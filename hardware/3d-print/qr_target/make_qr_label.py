#!/usr/bin/env python3
"""SEMES 와 같은 QR 플레이트 인쇄용 PDF 생성 (A4 가로, 실제 크기, 2쪽).

SEMES center.bmp 에서 실측한 값으로 만든다.
- QR 내용 FP11L / FP11C / FP11R, 버전 1 (21x21), 오류정정 M
  (libqrencode 결과가 SEMES QR 과 21x21 모듈 전부 일치)
- 모듈 0.545 mm (QR 11.45 mm), 밝은 바탕 13 mm, QR 중심 간격 18 mm, 플레이트 66 x 15 mm
- 카메라 화면 범위: 3840 / 14.25 px/mm x 2160 / 15.2 px/mm = 269.5 x 142.1 mm (모서리 표시)

1쪽: SEMES 와 같은 플레이트만
2쪽: 플레이트 + 화면 전체 + 표시 (평평한 벽에서 Y/Θ 구분용)

사용: python3 make_qr_target.py [-o qr_target_A4.pdf]
필요: 시스템 libqrencode (libqrencode.so.4)
"""
import argparse
import ctypes

from PIL import Image, ImageDraw, ImageFont

DPI = 600
A4_MM = (297.0, 210.0)  # 가로
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

QR_TEXTS = ("FP11L", "FP11C", "FP11R")
QR_MODULE_MM = 0.545
QR_LABEL_MM = 13.0
QR_PITCH_MM = 18.0
PLATE_MM = (69.0, 15.0)   # 2026-11-06 SEMES center.bmp 실측 (기존 66)
PLATE_RADIUS_MM = 3.0

# SEMES 3840x2160 사진 한 장에 찍히는 벽 위 범위 (가로 14.25 px/mm, 세로 x16/15)
FOV_MM = (3840 / 14.25, 2160 / (14.25 * 16 / 15))

GRID_PITCH_MM = 20.0
GRID_ARM_MM = 2.0
LINE_MM = 0.4


class _QRcode(ctypes.Structure):
    _fields_ = [("version", ctypes.c_int), ("width", ctypes.c_int),
                ("data", ctypes.POINTER(ctypes.c_ubyte))]


def qr_matrix(text):
    lib = ctypes.CDLL("libqrencode.so.4")
    lib.QRcode_encodeString.restype = ctypes.POINTER(_QRcode)
    lib.QRcode_encodeString.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int]
    lib.QRcode_free.argtypes = [ctypes.POINTER(_QRcode)]
    # version 1, QR_ECLEVEL_M = 1, QR_MODE_8 = 2, 대소문자 구분
    ptr = lib.QRcode_encodeString(text.encode(), 1, 1, 2, 1)
    qr = ptr.contents
    w = qr.width
    m = [[qr.data[y * w + x] & 1 for x in range(w)] for y in range(w)]
    lib.QRcode_free(ptr)
    return m


def mm2px(mm):
    return int(round(mm / 25.4 * DPI))


def rect(d, x0, y0, x1, y1, fill):
    d.rectangle([mm2px(x0), mm2px(y0), mm2px(x1) - 1, mm2px(y1) - 1], fill=fill)


def draw_plate(d, cx, cy):
    pw, ph = PLATE_MM
    d.rounded_rectangle([mm2px(cx - pw / 2), mm2px(cy - ph / 2), mm2px(cx + pw / 2), mm2px(cy + ph / 2)],
                        radius=mm2px(PLATE_RADIUS_MM), fill=0)
    for i, text in enumerate(QR_TEXTS):
        qx = cx + (i - 1) * QR_PITCH_MM
        rect(d, qx - QR_LABEL_MM / 2, cy - QR_LABEL_MM / 2, qx + QR_LABEL_MM / 2, cy + QR_LABEL_MM / 2, 255)
        m = qr_matrix(text)
        n = len(m)
        x0 = qx - n * QR_MODULE_MM / 2
        y0 = cy - n * QR_MODULE_MM / 2
        for r in range(n):
            for c in range(n):
                if m[r][c]:
                    rect(d, x0 + c * QR_MODULE_MM, y0 + r * QR_MODULE_MM,
                         x0 + (c + 1) * QR_MODULE_MM, y0 + (r + 1) * QR_MODULE_MM, 0)


def draw_fov_marks(d, cx, cy):
    fw, fh = FOV_MM
    arm = 10.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            x = cx + sx * fw / 2
            y = cy + sy * fh / 2
            # 모서리 꺾쇠: 화면 안쪽으로 뻗는다
            rect(d, min(x, x - sx * arm), y - LINE_MM / 2, max(x, x - sx * arm), y + LINE_MM / 2, 0)
            rect(d, x - LINE_MM / 2, min(y, y - sy * arm), x + LINE_MM / 2, max(y, y - sy * arm), 0)


def draw_cross(d, x, y):
    rect(d, x - GRID_ARM_MM, y - LINE_MM / 2, x + GRID_ARM_MM, y + LINE_MM / 2, 0)
    rect(d, x - LINE_MM / 2, y - GRID_ARM_MM, x + LINE_MM / 2, y + GRID_ARM_MM, 0)


def draw_grid(d, cx, cy):
    fw, fh = FOV_MM
    keep_out = (PLATE_MM[0] / 2 + 8, PLATE_MM[1] / 2 + 8)  # 플레이트 주변은 비워 둔다
    nx = int((fw / 2 - 5) // GRID_PITCH_MM)
    ny = int((fh / 2 - 5) // GRID_PITCH_MM)
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            x, y = cx + i * GRID_PITCH_MM, cy + j * GRID_PITCH_MM
            if abs(x - cx) < keep_out[0] and abs(y - cy) < keep_out[1]:
                continue
            draw_cross(d, x, y)


def draw_notes(d, title):
    font = ImageFont.truetype(FONT, mm2px(2.8))
    x0 = (A4_MM[0] - FOV_MM[0]) / 2
    # 100 mm 눈금자 (인쇄 배율 확인용)
    ry = 8.0
    rect(d, x0, ry, x0 + 100, ry + LINE_MM, 0)
    for i in range(11):
        tick = 3 if i % 5 == 0 else 1.5
        rect(d, x0 + i * 10, ry - tick, x0 + i * 10 + 0.3, ry, 0)
    d.text((mm2px(x0 + 104), mm2px(ry - 3)), "ruler = 100 mm  |  print at 100% (actual size)",
           font=font, fill=0)
    fw, fh = FOV_MM
    lines = [title,
             f"QR {'/'.join(QR_TEXTS)}, v1-M, module {QR_MODULE_MM} mm, label {QR_LABEL_MM:g} mm, "
             f"pitch {QR_PITCH_MM:g} mm, plate {PLATE_MM[0]:g}x{PLATE_MM[1]:g} mm",
             f"corner marks = camera view {fw:.1f} x {fh:.1f} mm (3840x2160 at 14.25 px/mm)"]
    y = A4_MM[1] - 16
    for line in lines:
        d.text((mm2px(x0), mm2px(y)), line, font=font, fill=0)
        y += 4.2


def make_page(with_grid):
    img = Image.new("1", (mm2px(A4_MM[0]), mm2px(A4_MM[1])), 1)
    d = ImageDraw.Draw(img)
    cx, cy = A4_MM[0] / 2, A4_MM[1] / 2
    draw_fov_marks(d, cx, cy)
    if with_grid:
        draw_grid(d, cx, cy)
    draw_plate(d, cx, cy)
    draw_notes(d, "Page 2: plate + background grid (20 mm) for Y/Theta separation on a flat wall"
               if with_grid else "Page 1: SEMES plate only")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="qr_target_A4.pdf")
    args = ap.parse_args()
    pages = [make_page(False), make_page(True)]
    pages[0].save(args.output, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    print(f"저장: {args.output}  (2쪽, 화면 범위 {FOV_MM[0]:.1f} x {FOV_MM[1]:.1f} mm)")


if __name__ == "__main__":
    main()

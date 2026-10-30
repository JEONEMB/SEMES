#!/usr/bin/env python3
"""카메라 캘리브레이션용 체커보드 PDF 생성 (A4 가로, 실제 크기).

기본값: 12 x 9 칸, 한 칸 15 mm -> 내부 코너 11 x 8 (보드 180 x 135 mm)
30 cm 거리에서 화면의 약 57% 를 차지해, 보드를 화면 구석까지 옮겨 가며 찍을 수 있다.
인쇄할 때 반드시 "실제 크기(100%)"로 인쇄하고, 아래 100 mm 눈금자로 확인한다.

사용: python3 make_checkerboard.py [--cols 12] [--rows 9] [--square 15] [-o 파일.pdf]
"""
import argparse

from PIL import Image, ImageDraw, ImageFont

DPI = 600
A4_MM = (297.0, 210.0)  # 가로
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def mm2px(mm):
    return int(round(mm / 25.4 * DPI))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=12, help="가로 칸 수")
    ap.add_argument("--rows", type=int, default=9, help="세로 칸 수")
    ap.add_argument("--square", type=float, default=15.0, help="한 칸 크기 (mm)")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    board_w, board_h = args.cols * args.square, args.rows * args.square
    if board_w > A4_MM[0] - 20 or board_h > A4_MM[1] - 24:
        raise SystemExit(f"보드 {board_w}x{board_h} mm 가 A4 가로에 들어가지 않음")

    img = Image.new("1", (mm2px(A4_MM[0]), mm2px(A4_MM[1])), 1)
    d = ImageDraw.Draw(img)

    # 보드는 가운데, 아래 여백에 눈금자와 설명을 둔다
    x0 = (A4_MM[0] - board_w) / 2
    y0 = (A4_MM[1] - board_h) / 2 - 4
    for r in range(args.rows):
        for c in range(args.cols):
            if (r + c) % 2 == 0:
                d.rectangle([mm2px(x0 + c * args.square), mm2px(y0 + r * args.square),
                             mm2px(x0 + (c + 1) * args.square) - 1,
                             mm2px(y0 + (r + 1) * args.square) - 1], fill=0)

    # 100 mm 눈금자 (10 mm 간격)
    ry = y0 + board_h + 5
    d.rectangle([mm2px(x0), mm2px(ry), mm2px(x0 + 100), mm2px(ry + 0.4)], fill=0)
    for i in range(11):
        tick = 3 if i % 5 == 0 else 1.5
        d.rectangle([mm2px(x0 + i * 10), mm2px(ry - tick), mm2px(x0 + i * 10 + 0.3), mm2px(ry)], fill=0)

    font = ImageFont.truetype(FONT, mm2px(3))
    text = (f"{args.cols}x{args.rows} squares, {args.square:g} mm, inner corners "
            f"{args.cols - 1}x{args.rows - 1}  |  print at 100% (actual size)  |  ruler = 100 mm")
    d.text((mm2px(x0 + 105), mm2px(ry - 2.5)), text, font=font, fill=0)

    out = args.output or f"checkerboard_A4_{args.cols}x{args.rows}_{args.square:g}mm.pdf"
    img.save(out, "PDF", resolution=DPI)
    print(f"저장: {out}  (보드 {board_w:g}x{board_h:g} mm, 내부 코너 {args.cols - 1}x{args.rows - 1})")


if __name__ == "__main__":
    main()

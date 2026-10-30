"""Y 사진(위 줄, 주황 테두리)과 Θ 사진(아래 줄, 초록 테두리)을 실제 사진 정사각형으로 놓고,
QR 위치가 같아 구분 안 되는 짝을 선으로 잇는다. 모든 사진에서 같은 위치(QR 판 주변)를 잘라 판이 움직인 게 보이게 한다."""
import numpy as np, pandas as pd, cv2, sys
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
P = Path(__file__).parent.parent
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
FONT = lambda s: ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf', s)
FONTB = lambda s: ImageFont.truetype(r'C:\Windows\Fonts\malgunbd.ttf', s)
ORG, GRN, BLUE, RED, TXT, SUB = (237, 125, 49), (112, 173, 71), (31, 78, 120), (192, 0, 0), (31, 31, 30), (95, 94, 88)
q = pd.read_csv(P/'14_데이터_QR마커위치.csv'); c0 = q[q.file == 'center.bmp'].iloc[0]
CX, CY, HALF = int(c0.Cx), int(c0.Cy), 450                                 # QR 판 주변 900x900 정사각형
def path(f): return ROOT/'center.bmp' if f == 'center.bmp' else ROOT/('Theta' if f[0] == 'T' else 'Y')/f
_cache = {}
def crop(f, size):
    if f not in _cache:
        im = cv2.imdecode(np.fromfile(str(path(f)), np.uint8), cv2.IMREAD_GRAYSCALE)
        _cache[f] = im[CY - HALF:CY + HALF, CX - HALF:CX + HALF]
    return Image.fromarray(cv2.resize(_cache[f], (size, size), interpolation=cv2.INTER_AREA))
# 판정 결과 ------------------------------------------------------------------
S = 'C:/Users/SSAFY/AppData/Local/Temp/claude/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES/0478c349-0ea5-4fa5-b878-4da27eb190c2/scratchpad/'
Yo, To = pd.read_pickle(S+'Yo.pkl'), pd.read_pickle(S+'To.pkl')
J = pd.read_pickle(P/'analysis_v2/judge.pkl'); r2 = J['res']['v2-a2 (판 전체 원근 변환 14개)']; Ys, Ts = J['Ys'], J['Ts']
def twins(yflag, tflag):
    tm, tv = Ts.m.values, Ts.value.values; ym, yv = Ys.m.values, Ys.value.values
    return ({round(v, 2): float(np.interp(m, tm, tv)) for v, m, f in zip(yv, ym, yflag) if f},
            {round(v, 2): float(np.interp(m, ym, yv)) for v, m, f in zip(tv, tm, tflag) if f})
PANELS = [('v1: QR 네모 가로 위치만', *twins((Ys.file.map(dict(zip(Yo.파일명, Yo.판정))) == '구분 불가').values,
                                          (Ts.file.map(dict(zip(To.파일명, To.판정))) == '구분 불가').values)),
          ('v2-a2: QR 판 전체 원근 변환 (QR만 쓰는 방법 중 최고)', *twins((r2['out']['Y'].규칙2 == '불가').values, (r2['out']['Θ'].규칙2 == '불가').values))]
fname = lambda ax, v: 'center.bmp' if v == 0 else f'{ax}{v:+.2f}.bmp'
yv = [round(v, 2) for v in np.r_[np.arange(-2, 0, 0.05), 0, np.arange(0.05, 2.001, 0.05)]]
def draw(tlim, tsz, out):
    tv = [round(v, 2) for v in np.r_[np.arange(-1, 0, 0.01), 0, np.arange(0.01, 1.0001, 0.01)] if abs(v) <= tlim + 1e-9]
    gap = max(2, tsz // 12); W = len(tv) * (tsz + gap) + 2 * 60
    ysz = min(int((W - 120) / len(yv)) - gap, 3 * tsz)
    xT = lambda v: 60 + (v + tlim) / (2 * tlim) * (len(tv) - 1) * (tsz + gap) + tsz / 2
    xY = lambda v: 60 + (v + 2) / 4 * (len(yv) - 1) * (ysz + gap) + ysz / 2 + ((W - 120) - len(yv) * (ysz + gap)) / 2
    HEAD = 60 + max(30, tsz // 2) + max(20, tsz // 3) + 60; PH = ysz + tsz + 520 + max(28, tsz // 2) + max(18, ysz // 4); Hh = HEAD + len(PANELS) * PH + 60
    img = Image.new('RGB', (W, Hh), 'white'); d = ImageDraw.Draw(img)
    d.text((60, 30), f'QR 위치만으로 구분 안 되는 Y 사진과 Θ 사진 (위 줄 = Y 사진 {len(yv)-1}장 + center, 아래 줄 = Θ 사진 {len(tv)-1}장 + center'
           + ('' if tlim >= 1 else f', Θ는 짝이 생길 수 있는 ±{tlim:.2f}만 표시') + ')', font=FONTB(max(30, tsz // 2)), fill=TXT)
    d.text((60, 30 + max(30, tsz // 2) + 14), '굵은 테두리 = 구분 안 됨, 흐린 사진 = 구분됨. 파란 실선 = Y 사진 → QR 위치가 같은 Θ 지점, 빨간 점선 = Θ 사진 → QR 위치가 같은 Y 지점. 모든 사진은 같은 위치(QR 판 주변 900x900px)를 자른 것.',
           font=FONT(max(20, tsz // 3)), fill=SUB)
    for k, (title, YT, TY) in enumerate(PANELS):
        top = HEAD + k * PH; yT = top + 40 + max(28, tsz // 2) + max(18, ysz // 4) + 30; yB = yT + ysz + 330
        d.text((60, top + 20), f'{title}  →  Y 쪽 {len(YT)}세트, Θ 쪽 {len(TY)}세트', font=FONTB(max(28, tsz // 2)), fill=TXT)
        for a, b in YT.items():
            if abs(b) <= tlim: d.line([(xY(a), yT + ysz + 6), (xT(b), yB - 6)], fill=BLUE, width=max(2, tsz // 25))
        for a, b in TY.items():
            if abs(a) > tlim: continue
            x0, y0, x1, y1 = xT(a), yB - 6, xY(b), yT + ysz + 6; n = 40
            for s in range(0, n, 2):
                d.line([(x0 + (x1 - x0) * s / n, y0 + (y1 - y0) * s / n), (x0 + (x1 - x0) * (s + 1) / n, y0 + (y1 - y0) * (s + 1) / n)], fill=RED, width=max(1, tsz // 35))
        for row, vals, sz, xf, ax, col, bad in [(yT, yv, ysz, xY, 'Y', ORG, YT), (yB, tv, tsz, xT, 'T', GRN, TY)]:
            for v in vals:
                ph = crop(fname(ax, v), sz).convert('RGB'); is_bad = v in bad; x = int(xf(v) - sz / 2)
                if not is_bad and v != 0: ph = Image.blend(ph, Image.new('RGB', ph.size, 'white'), 0.6)
                img.paste(ph, (x, int(row)))
                bw = max(3, sz // 14) if (is_bad or v == 0) else max(1, sz // 40)
                d.rectangle([x - bw, row - bw, x + sz + bw - 1, row + sz + bw - 1], outline=(TXT if v == 0 else col), width=bw)
            step = 4 if ax == 'Y' else (10 if tlim >= 1 else 5)
            for i, v in enumerate(vals):
                if i % step == 0 or v == 0:
                    lab = 'center' if v == 0 else (f'Y {v:+.2f}' if ax == 'Y' else f'Θ {v:+.2f}')
                    tw = d.textlength(lab, font=FONT(max(18, sz // 4)))
                    d.text((xf(v) - tw / 2, (row - max(18, sz // 4) - 14) if ax == 'Y' else (row + sz + 10)), lab, font=FONT(max(18, sz // 4)), fill=SUB)
        ylo, yhi = min(YT), max(YT); tl = list(YT.values())
        tw = d.textlength(f'{title}  →  Y 쪽 {len(YT)}세트, Θ 쪽 {len(TY)}세트', font=FONTB(max(28, tsz // 2)))
        d.text((60 + tw + 60, top + 20), f'구분 안 되는 Y: {ylo:+.2f} ~ {yhi:+.2f}mm   ↔   짝이 되는 Θ: {min(tl):+.3f} ~ {max(tl):+.3f}', font=FONTB(max(28, tsz // 2)), fill=BLUE)
    img.save(out, optimize=True); print(out.name, img.size)
draw(0.40, 150, P/'28_구분안되는짝_사진연결도_확대.png')
draw(1.00, 64, P/'28_구분안되는짝_사진연결도_전체.png')

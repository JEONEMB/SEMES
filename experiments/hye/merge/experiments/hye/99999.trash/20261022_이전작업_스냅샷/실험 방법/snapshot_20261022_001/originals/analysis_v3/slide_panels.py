"""슬라이드용 연결도 패널: 위 줄 Y 사진 81장(주황 테두리), 아래 줄 Θ 사진 81장(Θ ±0.40, 초록 테두리), 구분 안 되는 짝을 선으로."""
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from slide_data import *
OUT = HERE/'slide_img'; OUT.mkdir(exist_ok=True)
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
FONT = lambda s: ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf', s)
ORG, GRN, BLUE, RED, TXT, SUB = (237, 125, 49), (112, 173, 71), (31, 78, 120), (192, 0, 0), (31, 31, 30), (95, 94, 88)
c0 = q[q.file == 'center.bmp'].iloc[0]; CX, CY, HALF = int(c0.Cx), int(c0.Cy), 450
_cache = {}
def crop(f, size):
    if f not in _cache:
        p = ROOT/'center.bmp' if f == 'center.bmp' else ROOT/('Theta' if f[0] == 'T' else 'Y')/f
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        _cache[f] = cv2.resize(im[CY - HALF:CY + HALF, CX - HALF:CX + HALF], (96, 96), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.resize(_cache[f], (size, size), interpolation=cv2.INTER_AREA)).convert('RGB')
fname = lambda ax, v: 'center.bmp' if v == 0 else f'{ax}{v:+.2f}.bmp'
YV = [round(v, 2) for v in np.r_[np.arange(-2, 0, 0.05), 0, np.arange(0.05, 2.001, 0.05)]]
TV = [round(v, 2) for v in np.r_[np.arange(-0.40, 0, 0.01), 0, np.arange(0.01, 0.4001, 0.01)]]

def draw_panel(d, out, sz=44, gap=4, mid=170):
    n = len(YV); W = 30 + n * (sz + gap) + 30; H = 34 + sz + mid + sz + 40
    img = Image.new('RGB', (W, H), 'white'); dr = ImageDraw.Draw(img)
    x = lambda i: 30 + i * (sz + gap) + sz / 2
    xY = lambda v: x((v + 2) / 0.05); xT = lambda v: x((v + 0.40) / 0.01)
    yT, yB = 34, 34 + sz + mid
    for f, b in d['twinY'].items():
        if d['flagY'].get(f) == '불가' and abs(b) <= 0.40: dr.line([(xY(val(f)), yT + sz + 4), (xT(b), yB - 4)], fill=BLUE, width=2)
    for f, b in d['twinT'].items():
        a = val(f)
        if d['flagT'].get(f) == '불가' and abs(a) <= 0.40:
            x0, y0, x1, y1 = xT(a), yB - 4, xY(b), yT + sz + 4
            for s in range(0, 30, 2):
                dr.line([(x0 + (x1 - x0) * s / 30, y0 + (y1 - y0) * s / 30), (x0 + (x1 - x0) * (s + 1) / 30, y0 + (y1 - y0) * (s + 1) / 30)], fill=RED, width=1)
    for row, vals, ax, col, fl, xf in [(yT, YV, 'Y', ORG, d['flagY'], xY), (yB, TV, 'T', GRN, d['flagT'], xT)]:
        for v in vals:
            f = fname(ax, v); bad = fl.get(f) == '불가'; ph = crop(f, sz)
            if not bad and v != 0: ph = Image.blend(ph, Image.new('RGB', ph.size, 'white'), 0.65)
            X0 = int(xf(v) - sz / 2); img.paste(ph, (X0, row))
            bw = 4 if (bad or v == 0) else 1
            dr.rectangle([X0 - bw, row - bw, X0 + sz + bw - 1, row + sz + bw - 1], outline=(TXT if v == 0 else col), width=bw)
        for v in vals[::10] + ([0] if 0 not in vals[::10] else []):
            lab = 'center' if v == 0 else (f'Y {v:+.1f}' if ax == 'Y' else f'Θ {v:+.2f}')
            tw = dr.textlength(lab, font=FONT(20))
            dr.text((xf(v) - tw / 2, row - 30 if ax == 'Y' else row + sz + 8), lab, font=FONT(20), fill=SUB)
    img.save(out, optimize=True); return img.size

if __name__ == '__main__':
    for b in list(BASES) + ['최적']:
        print(b, draw_panel(pairing(b) if b != '최적' else pairing_best(), OUT/f'basis_{b}.png'))
    for k, d in methods().items():
        print(k, draw_panel(d, OUT/f'method_{k}.png'))
    # 범례용 작은 예시
    print('done')

"""Labeled stills of confusable Y/Θ pairs (v7 final predictions),
same look as make_ar2_label.py: green box = QR area at center, red box = tracked QR area, red value."""
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(r'C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set')
OUT = Path(__file__).parent
SQUARE = '--square' in sys.argv   # 1000x1000 crop centered on the plate, files end in _sq
if SQUARE:
    CROP = (640, 1640, 1402, 2402)
    QR_BOX = (140, 320, 860, 536)
else:
    CROP = (690, 1530, 1220, 2620)
    QR_BOX = (322, 270, 1042, 486)
SEARCH = 120
GREEN, RED = (40, 210, 60), (230, 30, 40)   # RGB
BOX_W = 7
FONT = ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', 120)
TEXT_Y = 960 if SQUARE else 805
PHOTOS = [  # confusable Y/Θ pairs (similar QR shift); earlier: Y-0.30 / T-0.05
    ('Y', 'Y+0.15', 'Y : +0.15mm'), ('Theta', 'T+0.03', 'Θ : +0.03'),
    ('Y', 'Y+0.20', 'Y : +0.20mm'), ('Theta', 'T+0.04', 'Θ : +0.04'),
    ('Y', 'Y+0.25', 'Y : +0.25mm'), ('Theta', 'T+0.05', 'Θ : +0.05'),
    ('Y', 'Y+0.50', 'Y : +0.50mm'), ('Theta', 'T+0.09', 'Θ : +0.09'),
]

def read(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    y0, y1, x0, x1 = CROP
    return cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

def shift(gray, template):
    x0, y0, x1, y1 = QR_BOX
    roi = gray[y0 - SEARCH:y1 + SEARCH, x0 - SEARCH:x1 + SEARCH]
    _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))
    if score < .8:
        raise RuntimeError(f'Unreliable QR match: {score:.3f}')
    return loc[0] - SEARCH

def box(draw, dx, color):
    x0, y0, x1, y1 = QR_BOX
    draw.rectangle((x0 + dx, y0, x1 + dx, y1), outline=color, width=BOX_W)

x0, y0, x1, y1 = QR_BOX
template = cv2.cvtColor(read(ROOT / 'center.bmp'), cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1]
for (sub, name, label), with_text in [(p, t) for p in PHOTOS for t in (True, False)]:
    s = label if with_text else ''
    rgb = read(ROOT / sub / f'{name}.bmp')   # _label.png with value text, _box.png boxes only
    dx = shift(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), template)
    print(name, 'dx', dx)
    pil = Image.fromarray(rgb)
    shadow = Image.new('L', pil.size, 0)
    sd = ImageDraw.Draw(shadow)
    box(sd, 0, 255)
    box(sd, dx, 255)
    tx = (pil.width - sd.textlength(s, font=FONT)) / 2
    sd.text((tx, TEXT_Y), s, font=FONT, fill=255, anchor='ls', stroke_width=6, stroke_fill=255)
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    pil = Image.composite(Image.new('RGB', pil.size, (0, 0, 0)), pil, shadow.point(lambda v: v * 0.55))
    d = ImageDraw.Draw(pil)
    box(d, 0, GREEN)
    box(d, dx, RED)
    d.text((tx, TEXT_Y), s, font=FONT, fill=RED, anchor='ls', stroke_width=4, stroke_fill=(255, 255, 255))
    pil.save(OUT / f"confuse_{name}_{'label' if with_text else 'box'}{'_sq' if SQUARE else ''}.png")

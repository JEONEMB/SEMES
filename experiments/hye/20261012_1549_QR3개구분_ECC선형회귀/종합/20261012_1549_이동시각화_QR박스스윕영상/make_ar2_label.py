"""Presentation version of AR2.mp4 (= Y_sweep_QR_zoom_boxes_redtop_x4): Y 0 -> +2.00 photos,
0.1 step, green box fixed at the center photo, red box tracks the QRs in front.
Bottom label 0mm (green) -> +2.00mm (red) matches the photos.
--theta: Theta photos T 0 -> +1.00 (0.05 step), so the green box sits at the center photo;
label 0 -> +1.00 matches the photos (unit unconfirmed, none shown)."""
from pathlib import Path
import subprocess
import sys
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(r'C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set')
THETA = '--theta' in sys.argv
OUT = Path(__file__).parent / ('AR2_theta_label.mp4' if THETA else 'AR2_label.mp4')
FPS, HOLD = 40, 20               # same timing as AR2: 0.5 s green-only, then 20 photos
CROP = (690, 1530, 1220, 2620)
QR_BOX = (322, 270, 1042, 486)
SEARCH = 120
GREEN, RED = (40, 210, 60), (230, 30, 40)   # RGB
BOX_W = 7
FONT = ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', 120)
TEXT_Y = 805

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
    return loc[0] - SEARCH, loc[1] - SEARCH

def box(draw, dx, color):
    x0, y0, x1, y1 = QR_BOX
    draw.rectangle((x0 + dx, y0, x1 + dx, y1), outline=color, width=BOX_W)

def render(i, show_red, value):
    pil = Image.fromarray(frames[i])
    if THETA:
        s = 'Θ : 0' if value == 0 else f'Θ : +{value:.2f}'
    else:
        s = 'Y : 0mm' if value == 0 else f'Y : +{value:.2f}mm'
    color = GREEN if value == 0 else RED
    # soft dark shadow behind boxes and text so they pop on the light metal
    shadow = Image.new('L', pil.size, 0)
    sd = ImageDraw.Draw(shadow)
    box(sd, shifts[0][0], 255)
    if show_red:
        box(sd, shifts[i][0], 255)
    w = sd.textlength(s, font=FONT)
    tx = (pil.width - w) / 2
    sd.text((tx, TEXT_Y), s, font=FONT, fill=255, anchor='ls', stroke_width=6, stroke_fill=255)
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    pil = Image.composite(Image.new('RGB', pil.size, (0, 0, 0)), pil, shadow.point(lambda v: v * 0.55))
    d = ImageDraw.Draw(pil)
    box(d, shifts[0][0], GREEN)
    if show_red:
        box(d, shifts[i][0], RED)
    d.text((tx, TEXT_Y), s, font=FONT, fill=color, anchor='ls', stroke_width=4, stroke_fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

if THETA:   # T = k * 0.01, k = 0..100 step 5, 21 photos, starts at center
    paths = [ROOT / 'center.bmp' if k == 0 else ROOT / 'Theta' / f'T{k / 100:+.2f}.bmp' for k in range(0, 101, 5)]
else:       # Y = k * 0.05, k = 0..40 step 2, 21 photos, starts at center
    paths = [ROOT / 'center.bmp' if k == 0 else ROOT / 'Y' / f'Y{k * 0.05:+.2f}.bmp' for k in range(0, 41, 2)]
x0, y0, x1, y1 = QR_BOX
frames = [read(p) for p in paths]
template = cv2.cvtColor(read(ROOT / 'center.bmp'), cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1]
shifts = [shift(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), template) for f in frames]
values = [n / (20 if THETA else 10) for n in range(len(frames))]   # displayed 0 -> 1.00 / 2.00
print('shifts', shifts[0], shifts[-1])

order = [(0, False)] * HOLD + [(i, True) for i in range(1, len(frames))]
if '--check' in sys.argv:
    cv2.imwrite(str(Path(sys.argv[-1], 'ar2_theta_check.png' if THETA else 'ar2_check.png')),
                np.vstack([render(0, False, 0), render(len(frames) - 1, True, values[-1])]))
else:
    with tempfile.TemporaryDirectory() as tmp:
        for n, (i, red) in enumerate(order):
            cv2.imwrite(str(Path(tmp, f'{n:03d}.png')), render(i, red, values[i]))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                        '-i', str(Path(tmp, '%03d.png')), '-c:v', 'libx264', '-profile:v', 'main',
                        '-level', '4.1', '-crf', '14', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(OUT)], check=True)

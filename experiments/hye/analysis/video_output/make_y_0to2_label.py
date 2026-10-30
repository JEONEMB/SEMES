"""Y 0 -> +2.00 mm (0.1 step) QR zoom video: green box = QR area at Y 0 (fixed),
red box = tracked QR area (drawn in front), Y value at the bottom (green at 0, red otherwise)."""
from pathlib import Path
import subprocess
import sys
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set')
OUT = Path(__file__).parent / 'Y_0to2_QR_zoom_boxes_label.mp4'
FPS = 10                         # one photo per frame, 0.1 s each
HOLD_START, HOLD_END = 5, 5      # frames: green-only start, final value hold
CROP = (690, 1530, 1220, 2620)   # y0, y1, x0, x1, same as make_y_sweep_clean.py
QR_BOX = (322, 270, 1042, 486)   # x0, y0, x1, y1 of the 3-QR area, crop coords, center photo
SEARCH = 120
GREEN, RED = (60, 200, 60), (40, 40, 230)   # BGR
FONT = ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', 110)
TEXT_Y = 805                     # baseline area near the bottom of the 840 px frame

def read(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    y0, y1, x0, x1 = CROP
    return im[y0:y1, x0:x1]

def shift(gray, template):
    x0, y0, x1, y1 = QR_BOX
    roi = gray[y0 - SEARCH:y1 + SEARCH, x0 - SEARCH:x1 + SEARCH]
    _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))
    if score < .8:
        raise RuntimeError(f'Unreliable QR match: {score:.3f}')
    return loc[0] - SEARCH, loc[1] - SEARCH

def rect(im, dx, dy, color):
    x0, y0, x1, y1 = QR_BOX
    cv2.rectangle(im, (x0 + dx, y0 + dy), (x1 + dx, y1 + dy), color, 4, cv2.LINE_AA)

def label(im, value):
    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    s = 'Y : 0mm' if value == 0 else f'Y : {value:+.2f}mm'
    color = GREEN if value == 0 else RED   # green at zero, red for nonzero offsets
    w = draw.textlength(s, font=FONT)
    draw.text(((im.shape[1] - w) / 2, TEXT_Y), s, font=FONT, fill=color[::-1], anchor='ls')
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

values = [k / 10 for k in range(0, 21)]
paths = [ROOT / 'center.bmp' if v == 0 else ROOT / 'Y' / f'Y{v:+.2f}.bmp' for v in values]
x0, y0, x1, y1 = QR_BOX
frames = [read(p) for p in paths]
template = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
shifts = [shift(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), template) for f in frames]
print('shifts', shifts)

def frame(i, show_red):
    im = frames[i].copy()
    rect(im, *shifts[0], GREEN)
    if show_red:
        rect(im, *shifts[i], RED)
    return label(im, values[i])

order = [(0, False)] * HOLD_START + [(i, True) for i in range(1, len(frames))] + [(len(frames) - 1, True)] * HOLD_END
if '--check' in sys.argv:
    cv2.imwrite(str(Path(sys.argv[-1], 'label_check.png')),
                np.vstack([frame(0, False)[600:], frame(len(frames) - 1, True)[600:]]))
else:
    with tempfile.TemporaryDirectory() as tmp:
        for n, (i, red) in enumerate(order):
            cv2.imwrite(str(Path(tmp, f'{n:03d}.png')), frame(i, red))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                        '-i', str(Path(tmp, '%03d.png')), '-r', '30', '-c:v', 'libx264',
                        '-profile:v', 'main', '-level', '4.1', '-crf', '14', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(OUT)], check=True)

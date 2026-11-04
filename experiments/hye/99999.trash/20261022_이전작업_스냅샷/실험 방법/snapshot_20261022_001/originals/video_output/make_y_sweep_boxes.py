"""Y -2.00 -> +2.00 QR zoom video with boxes: green = QR area in the first frame (fixed),
red = same QR area tracked in each frame by template matching. No text."""
from pathlib import Path
import subprocess
import sys
import tempfile
import cv2
import numpy as np

ROOT = Path(r'C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set')
OUT = Path(__file__).parent
FPS = 40
HOLD = 20                        # frames (0.5 s) of the first photo with only the green box
CROP = (690, 1530, 1220, 2620)   # y0, y1, x0, x1, same as make_y_sweep_clean.py
QR_BOX = (322, 270, 1042, 486)   # x0, y0, x1, y1 of the 3-QR area, crop coords, center photo
SEARCH = 120                     # px search margin around the box
GREEN, RED = (60, 200, 60), (40, 40, 230)
RED_TOP = '--red-top' in sys.argv   # draw red over green (default: green over red)

def read(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    y0, y1, x0, x1 = CROP
    return im[y0:y1, x0:x1]

def sequence():
    for k in range(-40, 41):
        yield ROOT / 'center.bmp' if k == 0 else ROOT / 'Y' / f'Y{k * 0.05:+.2f}.bmp'

def shift(gray, template):
    x0, y0, x1, y1 = QR_BOX
    roi = gray[y0 - SEARCH:y1 + SEARCH, x0 - SEARCH:x1 + SEARCH]
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < .8:
        raise RuntimeError(f'Unreliable QR match: {score:.3f}')
    return loc[0] - SEARCH, loc[1] - SEARCH

def rect(im, dx, dy, color):
    x0, y0, x1, y1 = QR_BOX
    cv2.rectangle(im, (x0 + dx, y0 + dy), (x1 + dx, y1 + dy), color, 4, cv2.LINE_AA)

paths = list(sequence())
x0, y0, x1, y1 = QR_BOX
template = cv2.cvtColor(read(ROOT / 'center.bmp'), cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
frames = [read(p) for p in paths]
shifts = [shift(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), template) for f in frames]
print('first', shifts[0], 'last', shifts[-1])

def encode(idx, out):
    with tempfile.TemporaryDirectory() as tmp:
        idx = list(idx)
        for n, i in enumerate([idx[0]] * HOLD + idx[1:]):
            im = frames[i].copy()
            if RED_TOP:
                rect(im, *shifts[0], GREEN)
            if n >= HOLD:
                rect(im, *shifts[i], RED)
            if not RED_TOP:
                rect(im, *shifts[0], GREEN)
            cv2.imwrite(str(Path(tmp, f'{n:03d}.png')), im)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                        '-i', str(Path(tmp, '%03d.png')), '-c:v', 'libx264', '-profile:v', 'main',
                        '-level', '4.1', '-crf', '14', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(out)], check=True)

if '--check' in sys.argv:
    for i, name in [(0, 'first'), (len(frames) - 1, 'last')]:
        im = frames[i].copy()
        rect(im, *shifts[0], GREEN)
        rect(im, *shifts[i], RED)
        cv2.imwrite(str(Path(sys.argv[-1], f'box_{name}.png')), im)
else:
    tag = '_redtop' if RED_TOP else ''
    encode(range(len(frames)), OUT / f'Y_sweep_QR_zoom_boxes{tag}.mp4')          # 81 photos
    encode(range(0, len(frames), 4), OUT / f'Y_sweep_QR_zoom_boxes{tag}_x4.mp4')  # 21 photos

"""Clean Y -2.00 -> +2.00 sweep video (all 80 Y photos + center): raw photos only, no text or overlays."""
from pathlib import Path
import subprocess
import tempfile
import cv2
import numpy as np

ROOT = Path(r'C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set')
OUT = Path(__file__).parent
SEC_PER_PHOTO = 0.025  # 4x faster than 0.1 s, one photo per frame at 40 fps
CROP = (690, 1530, 1220, 2620)  # y0, y1, x0, x1 around the 3 QR markers

def read(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return im

def sequence():
    for k in range(-40, 41):
        v = k * 0.05
        yield ROOT / 'center.bmp' if k == 0 else ROOT / 'Y' / f'Y{v:+.2f}.bmp'

def encode(frames_dir, out, vf='null'):
    fps = 1 / SEC_PER_PHOTO
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', f'{fps}',
                    '-i', str(frames_dir / '%03d.png'), '-r', f'{fps}', '-vf', vf,
                    '-c:v', 'libx264', '-profile:v', 'main', '-level', '4.1', '-crf', '14',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], check=True)

with tempfile.TemporaryDirectory() as tmp:
    full_dir, crop_dir = Path(tmp, 'full'), Path(tmp, 'crop')
    full_dir.mkdir(); crop_dir.mkdir()
    y0, y1, x0, x1 = CROP
    for i, path in enumerate(sequence()):
        im = read(path)
        cv2.imwrite(str(full_dir / f'{i:03d}.png'), im)
        cv2.imwrite(str(crop_dir / f'{i:03d}.png'), im[y0:y1, x0:x1])
        print(path.name)
    encode(full_dir, OUT / 'Y_sweep_full_1080p.mp4', 'scale=1920:1080:flags=area')
    encode(crop_dir, OUT / 'Y_sweep_QR_zoom.mp4')

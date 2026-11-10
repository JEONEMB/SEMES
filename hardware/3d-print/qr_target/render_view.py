#!/usr/bin/env python3
"""조립 타깃을 SEMES 카메라(핀홀 fx 4450)로 찍은 모습을 그려 SEMES center.bmp와 나란히 비교한다."""
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

import make_target as T

SCALE = 0.25                       # 3840x2160 -> 960x540
W, H = int(3840 * SCALE), int(2160 * SCALE)


def project(v):
    zc = v[:, 0] + T.PUPIL
    u = T.CX + T.FX * (-v[:, 1]) / zc
    w = T.CY + T.FY * (T.LENS_H - v[:, 2]) / zc
    return np.stack([u * SCALE, w * SCALE], 1), zc


def render(meshes, light=np.array([-0.6, 0.15, 0.8])):   # 빛이 오는 쪽 (카메라 쪽 위)
    light = light / np.linalg.norm(light)
    img = Image.new("L", (W, H), 235)
    d = ImageDraw.Draw(img)
    tris = []
    for m in meshes:
        p, z = project(m.vertices)
        n = m.face_normals
        for f, nn in zip(m.faces, n):
            if nn[0] > 0:                    # 카메라 반대쪽 면
                continue
            shade = 60 + 190 * max(0.0, float(np.dot(nn, light)))
            tris.append((z[f].mean(), [tuple(p[i]) for i in f], int(min(250, shade))))
    tris.sort(key=lambda t: -t[0])
    for _, pts, s in tris:
        d.polygon(pts, fill=s)
    return img


def main():
    parts = T.main()
    meshes = [T.to_trimesh(m) for m in parts.values()]
    img = render(meshes).convert("RGB")
    d = ImageDraw.Draw(img)
    # SEMES 기준 위치 표시 (빨강): QR 중심, W2/W3·나사, 위/아래 가로선 구간
    red = (220, 40, 40)
    for (x, y) in [(1650, 1065), (1907, 1065), (2165, 1065)]:
        d.rectangle([(x - 93) * SCALE, (y - 100) * SCALE, (x + 93) * SCALE, (y + 100) * SCALE], outline=red)
    for (x, y) in [(960, 1352), (2867, 1357), (1010, 1677), (2820, 1684)]:
        d.ellipse([(x - 30) * SCALE, (y - 30) * SCALE, (x + 30) * SCALE, (y + 30) * SCALE], outline=red, width=2)
    for x0, x1, y in [(400, 1168, 716), (2700, 3468, 723), (400, 1168, 1788), (2600, 3368, 1797)]:
        d.line([x0 * SCALE, y * SCALE, x1 * SCALE, y * SCALE], fill=red, width=1)
    sem = Image.open(sys.argv[1]).convert("RGB").resize((W, H)) if len(sys.argv) > 1 else None
    out = Path(__file__).resolve().parent / "preview_camera_view.png"
    if sem:
        both = Image.new("RGB", (W, H * 2 + 10), (255, 255, 255))
        both.paste(sem, (0, 0))
        both.paste(img, (0, H + 10))
        both.save(out)
    else:
        img.save(out)
    print("저장:", out)


if __name__ == "__main__":
    main()

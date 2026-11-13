#!/usr/bin/env python3
"""간단 버전 타깃을 SEMES 카메라로 본 모습 (SEMES center.bmp와 비교)."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import make_target as T, make_target_board as S, render_view as R
parts = S.main()
img = R.render([T.to_trimesh(m) for m in parts.values()]).convert("RGB")
d = ImageDraw.Draw(img); red = (220, 40, 40); k = R.SCALE
for (x, y) in [(1650, 1065), (1907, 1065), (2165, 1065)]:
    d.rectangle([(x-93)*k, (y-100)*k, (x+93)*k, (y+100)*k], outline=red)
for (x, y) in [(960, 1352), (2867, 1357), (1010, 1677), (2820, 1684)]:
    d.ellipse([(x-30)*k, (y-30)*k, (x+30)*k, (y+30)*k], outline=red, width=2)
for x0, x1, y in [(400, 1168, 716), (2700, 3468, 723), (400, 1168, 1788), (2600, 3368, 1797)]:
    d.line([x0*k, y*k, x1*k, y*k], fill=red, width=1)
sem = Image.open(sys.argv[1]).convert("RGB").resize((R.W, R.H))
both = Image.new("RGB", (R.W, R.H*2+10), (255,255,255)); both.paste(sem, (0,0)); both.paste(img, (0, R.H+10))
out = Path(__file__).resolve().parent / "preview_camera_view.png"; both.save(out); print("저장:", out)

import os, json, itertools, numpy as np
from PIL import Image, ImageDraw, ImageFont
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set"
z = np.load(os.path.join(SC, "tracks.npz"), allow_pickle=True); X, D, pts = z["X"], z["M"][:, :, 0], z["pts"]
c = json.load(open(os.path.join(SC, "combos.json"))); info = c["info"]; good = c["good"]
EXCL = {"라벨": (1740, 1195, 2060, 1360), "QR 데이터": (1560, 975, 2250, 1160), "케이블": (1840, 0, 2260, 560)}
def why(k):
    if k == 0: return None                              # 파인더 패턴은 표준 모양 → 유지
    x, y = pts[k]
    for n, (a, b, cc, d) in EXCL.items():
        if a < x < cc and b < y < d: return n
    return None
excl = {k: why(k) for k in good if why(k)}
ok = [k for k in good if k not in excl]
print("제외", len(excl), {n: sum(1 for v in excl.values() if v == n) for n in EXCL}, "→ 남은 특징점", len(ok))
rms = lambda a: float(np.sqrt(np.mean(a**2))); idx = np.arange(len(X))
def ev(cols, tr, te):
    J = np.array([np.linalg.lstsq(X[tr], D[tr, k], rcond=None)[0] for k in cols])
    e = np.linalg.lstsq(J, D[te][:, cols].T, rcond=None)[0].T - X[te]; return rms(e[:, 0]), rms(e[:, 1]), np.linalg.cond(J)
fit, sel, fin = idx % 4 < 2, idx % 4 == 2, idx % 4 == 3
pairs = sorted(((ev([i, j], fit, sel)[1], i, j) for i, j in itertools.combinations(ok, 2)))
print("\n[새 최고 쌍 TOP 8]  (선택용 데이터로 고르고 → 처음 보는 데이터로 채점)")
for _, i, j in pairs[:8]:
    t, y, cn = ev([i, j], fit, fin)
    print(f"  #{i:2d}({pts[i][0]:.0f},{pts[i][1]:.0f}) b/a {info[i]['ratio']:.3f} + #{j:2d}({pts[j][0]:.0f},{pts[j][1]:.0f}) b/a {info[j]['ratio']:.3f}  →  Θ {t:.4f}  Y {y:.3f}mm  cond {cn:.0f}")
for name, cols in [("남은 전체 %d개" % len(ok), ok), ("기존 B (#1+#46, 라벨 포함)", [1, 46])]:
    t, y, cn = ev(cols, fit, fin); print(f"{name:28s} Θ {t:.4f}  Y {y:.3f}mm")
best = [pairs[0][1], pairs[0][2]]
# greedy (선택용 데이터 기준)
s = list(best); rest = [k for k in ok if k not in s]; hist = []
for _ in range(10):
    k = min(rest, key=lambda k: ev(s + [k], fit, sel)[1]); s.append(k); rest.remove(k); hist.append((len(s), *ev(s, fit, fin)[:2]))
print("greedy:", [(m, round(t, 4), round(y, 3)) for m, t, y in hist])
json.dump(dict(ok=ok, excl={int(k): v for k, v in excl.items()}, best=[int(b) for b in best]), open(os.path.join(SC, "stable.json"), "w"), ensure_ascii=False)
# 지도
im = Image.open(os.path.join(ROOT, "center.bmp")).convert("RGB"); dr = ImageDraw.Draw(im)
F = lambda s: ImageFont.truetype(r"C:\Windows\Fonts\malgunbd.ttf", s)
for n, (a, b, cc, d) in EXCL.items():
    dr.rectangle([a, b, cc, d], outline=(255, 60, 60), width=8); dr.text((a + 10, d + 8 if n != "케이블" else b + 570), f"제외: {n}", font=F(56), fill=(255, 60, 60), stroke_width=6, stroke_fill=(0, 0, 0))
for k in ok:
    x, y = pts[k]; dr.ellipse([x-16, y-16, x+16, y+16], outline=(60, 200, 255), width=6)
for k in excl:
    x, y = pts[k]; dr.line([x-14, y-14, x+14, y+14], fill=(255, 60, 60), width=6); dr.line([x-14, y+14, x+14, y-14], fill=(255, 60, 60), width=6)
for k in best:
    x, y = pts[k]; dr.ellipse([x-34, y-34, x+34, y+34], outline=(255, 230, 0), width=12)
    dr.text((x + 44, y - 40), f"#{k}", font=F(72), fill=(255, 230, 0), stroke_width=7, stroke_fill=(0, 0, 0))
im.resize((1920, 1080), Image.LANCZOS).save(r"C:\Users\SSAFY\Desktop\hwlee\projects\2026-10_SEMES\feature_map_stable.png")
im.resize((960, 540), Image.LANCZOS).save(os.path.join(SC, "fms.png"))

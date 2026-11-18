import sys, pathlib, numpy as np, cv2
sys.path.insert(0, "apps/distribution"); import reoof
S = pathlib.Path(sys.argv[1]); E = pathlib.Path("experiments/hye/20261112_1504_QR요약54개_536점대조/결과")
tf = reoof.tsv("training_features.tsv")
p4 = dict(zip([int(x) for x in (E / "round4_ids.txt").read_text().split()], [l.strip() for l in (E / "round4_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]))
rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001]
PINS = [(974, 1352), (2864, 1358), (1010, 1677), (2817, 1681)]; H = 70
def rd(p): return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
c = rd("data/clean/clean_20261014/center.bmp"); ref = np.array([c[y - H:y + H + 1, x - H:x + H + 1] for x, y in PINS])
crops = np.zeros((len(rows), 4, 2 * H + 1, 2 * H + 1), np.uint8); off = np.zeros((len(rows), 4, 2), int)
for i, r in enumerate(rows):
    rid = int(r["id"]); im = rd(p4[rid] if rid in p4 else r["path"]); f = np.array([float(v) for v in r["features66"].split(";")])[58:].reshape(4, 2)
    for k, (x, y) in enumerate(PINS):
        ox, oy = int(round(f[k, 0])), int(round(f[k, 1])); off[i, k] = (ox, oy); crops[i, k] = im[y + oy - H:y + oy + H + 1, x + ox - H:x + ox + H + 1]
    if i % 100 == 0: print(i, flush=True)
np.savez_compressed(S / "screw/crops.npz", ref=ref, crops=crops, off=off, ids=np.array([int(r["id"]) for r in rows]))
# 모양 확인용: 센터 + 세션별 원점에 가장 가까운 사진
src = np.array([r["source"] for r in rows]); L = np.array([[float(r["Y"]), float(r["T"])] for r in rows]); tiles = [np.hstack(list(ref))]
for s in sorted(set(src)):
    idx = np.where(src == s)[0]; j = idx[np.argmin(np.hypot(L[idx, 0] / .05, L[idx, 1] / .01))]; print(s, rows[j]["id"], L[j]); tiles.append(np.hstack(list(crops[j])))
cv2.imwrite(str(S / "screw/montage.png"), cv2.resize(np.vstack(tiles), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

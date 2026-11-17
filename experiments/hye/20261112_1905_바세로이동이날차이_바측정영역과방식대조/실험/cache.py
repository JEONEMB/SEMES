import sys, json, pathlib, numpy as np, cv2
sys.path.insert(0, "apps/distribution"); import reoof
S = pathlib.Path(sys.argv[1]); E = pathlib.Path("experiments/hye/20261112_1504_QR요약54개_536점대조/결과")
P = json.load(open(S / "f04/experiments/hye/20261111_0944_final_Y_T/assets/line_protocol.json", encoding="utf-8"))
for r in P["regions"]: print(r)
print({k: P["blocks"][0][k] for k in P["blocks"][0]}, len(P["blocks"]))
for b in P["blocks"][::12]: print(b["id"], b["x0"], b["x1"], b["reference_y"], b["read_box"])
tf = reoof.tsv("training_features.tsv")
p4 = dict(zip([int(x) for x in (E / "round4_ids.txt").read_text().split()], [l.strip() for l in (E / "round4_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]))
rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001]
def rd(p): return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
# 영역: (이름, x0, y0). 가로 1024(32 px 칸 32개), 세로 200
REG = []
for r in P["regions"]:
    ys = [b["reference_y"] for b in P["blocks"] if b["id"].startswith(r["name"])]; yc = int(round(np.mean(ys)))
    REG.append((r["name"], r["x_range"][0] - 128, yc - 100, yc))
print(REG)
def prof(im):
    out = np.zeros((len(REG), 32, 200), np.float32)
    for k, (_, x0, y0, _) in enumerate(REG):
        band = im[y0:y0 + 200, x0:x0 + 1024].astype(np.float32); out[k] = band.reshape(200, 32, 32).mean(2).T
    return out
c = rd("data/clean/clean_20261014/center.bmp"); ref = prof(c)
prof_all = np.zeros((len(rows), len(REG), 32, 200), np.float32)
for i, r in enumerate(rows):
    rid = int(r["id"]); prof_all[i] = prof(rd(p4[rid] if rid in p4 else r["path"]))
    if i % 100 == 0: print(i, flush=True)
np.savez_compressed(S / "bar/profiles.npz", ref=ref, prof=prof_all, ids=np.array([int(r["id"]) for r in rows]), reg=np.array([[x0, y0, yc] for _, x0, y0, yc in REG]), names=np.array([n for n, *_ in REG]))
# 모양 확인: 센터 사진의 네 영역(세로 3배 확대)
tiles = [cv2.resize(c[y0:y0 + 200, x0:x0 + 1024], (1024, 400), interpolation=cv2.INTER_CUBIC) for _, x0, y0, _ in REG]
cv2.imwrite(str(S / "bar/bands_center.png"), np.vstack(tiles))

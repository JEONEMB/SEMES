"""나사 4개의 이동을 위상 상관(cv2.phaseCorrelate)으로 다시 잰다.

비교 대상(지금 방식): 기울기 영상 ECC 맞춤. 값은 20261112_1756 실험 결과/meas.npz 의 "기울기영상_전체".
조각 위치는 지금 방식과 같다: 나사 기준 위치 + 00004 핀 관측 이동(정수 반올림), 141×141.

위상 상관 후보
- pc_raw      : 밝기 조각, Hann 창
- pc_grad     : 기울기 영상 조각(흐림 1.5 뒤 Sobel 크기, 지금 방식과 같은 전처리), Hann 창
- pc_raw_ni   : 초기값 없이 나사 기준 위치에서 바로 자른 257×257 밝기 조각, Hann 창
- pc_grad_ni  : 같은 조각의 기울기 영상
_ni 는 핀 관측(초기값)을 빼도 되는지 보려는 후보다.

출력: 결과/pc_meas.npz (후보마다 702×8, x,y 순서), C++ 속도 측정용 조각은 인자로 준 캐시 폴더에 raw 로 저장.
실행: 저장소 루트에서  python "<이 폴더>/실험/measure_pc.py" <캐시 폴더>
"""
import sys, pathlib, time, numpy as np, cv2
sys.path.insert(0, "apps/distribution"); import reoof

EXP = pathlib.Path(__file__).resolve().parent.parent; OUT = EXP / "결과"
CACHE = pathlib.Path(sys.argv[1]); CACHE.mkdir(parents=True, exist_ok=True)
E = pathlib.Path("experiments/hye/20261112_1504_QR요약54개_536점대조/결과")
tf = reoof.tsv("training_features.tsv")
p4 = dict(zip([int(x) for x in (E / "round4_ids.txt").read_text().split()], [l.strip() for l in (E / "round4_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]))
rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001]
PINS = [(974, 1352), (2864, 1358), (1010, 1677), (2817, 1681)]; H, HN = 70, 128


def rd(p): return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
def grad(a):
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))


c = rd("data/clean/clean_20261014/center.bmp")
refs = np.array([c[y - H:y + H + 1, x - H:x + H + 1] for x, y in PINS]); refs_ni = np.array([c[y - HN:y + HN + 1, x - HN:x + HN + 1] for x, y in PINS])
win, win_ni = cv2.createHanningWindow((2 * H + 1, 2 * H + 1), cv2.CV_64F), cv2.createHanningWindow((2 * HN + 1, 2 * HN + 1), cv2.CV_64F)
R = dict(raw=[r.astype(np.float64) for r in refs], grad=[grad(r).astype(np.float64) for r in refs],
         raw_ni=[r.astype(np.float64) for r in refs_ni], grad_ni=[grad(r).astype(np.float64) for r in refs_ni])
n = len(rows); out = {k: np.zeros((n, 4, 2)) for k in ("pc_raw", "pc_grad", "pc_raw_ni", "pc_grad_ni")}
resp = {k: np.zeros((n, 4)) for k in out}
crops = np.zeros((n, 4, 2 * H + 1, 2 * H + 1), np.uint8); crops_ni = np.zeros((n, 4, 2 * HN + 1, 2 * HN + 1), np.uint8); off = np.zeros((n, 4, 2), np.int32)
t0 = time.time()
for i, r in enumerate(rows):
    rid = int(r["id"]); im = rd(p4[rid] if rid in p4 else r["path"]); f = np.array([float(v) for v in r["features66"].split(";")])[58:].reshape(4, 2)
    for k, (x, y) in enumerate(PINS):
        ox, oy = int(round(f[k, 0])), int(round(f[k, 1])); off[i, k] = (ox, oy)
        a = im[y + oy - H:y + oy + H + 1, x + ox - H:x + ox + H + 1]; b = im[y - HN:y + HN + 1, x - HN:x + HN + 1]
        crops[i, k], crops_ni[i, k] = a, b
        for key, ref, img, w, base in (("pc_raw", R["raw"][k], a.astype(np.float64), win, (ox, oy)), ("pc_grad", R["grad"][k], grad(a).astype(np.float64), win, (ox, oy)),
                                       ("pc_raw_ni", R["raw_ni"][k], b.astype(np.float64), win_ni, (0, 0)), ("pc_grad_ni", R["grad_ni"][k], grad(b).astype(np.float64), win_ni, (0, 0))):
            (dx, dy), rs = cv2.phaseCorrelate(ref, img, w)
            out[key][i, k] = (base[0] + dx, base[1] + dy); resp[key][i, k] = rs
    if i % 100 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
ids = np.array([int(r["id"]) for r in rows])
np.savez(OUT / "pc_meas.npz", ids=ids, off=off, **{k: v.reshape(n, 8) for k, v in out.items()}, **{"resp_" + k: v for k, v in resp.items()})
refs.tofile(CACHE / "refs141.u8"); crops.tofile(CACHE / "crops141.u8"); refs_ni.tofile(CACHE / "refs257.u8"); crops_ni.tofile(CACHE / "crops257.u8"); off.tofile(CACHE / "off.i32")
print("저장", n, "장")

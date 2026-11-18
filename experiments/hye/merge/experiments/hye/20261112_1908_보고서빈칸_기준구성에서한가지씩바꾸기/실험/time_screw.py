"""나사 기울기 영상 맞춤(4개)의 사진 한 장 계산 시간. 20261112_1756 실험 measure.py 의 기울기영상_전체 와 같은 계산이다.
영상을 메모리에 올린 뒤부터 잰다. 시작 위치는 00004 의 나사 값(밝기 조각 측정)을 반올림해 쓴다. 1~3차 사진 중 30장(seed 20261112), 3번 돌려 뒤 2번의 최솟값.
실행: 저장소 루트에서  python "<이 폴더>/실험/time_screw.py" > "<이 폴더>/결과/나사속도_출력.txt"   (data/clean 의 BMP 가 필요하다)
"""
import sys, time, numpy as np, cv2, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, "apps/distribution"); import reoof
cv2.setNumThreads(1)
tf = reoof.tsv("training_features.tsv")
rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001 and r["source"] != "20261109_round4"]
rng = np.random.default_rng(20261112); pick = [rows[i] for i in rng.choice(len(rows), 30, replace=False)]
PINS = [(974, 1352), (2864, 1358), (1010, 1677), (2817, 1681)]; H = 70
yy, xx = np.mgrid[-H:H + 1, -H:H + 1]; full = (np.hypot(xx, yy) <= 45).astype(np.uint8)
def rd(p): return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
def grad(a): a = cv2.GaussianBlur(a, (0, 0), 1.5); return cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-8)
c = rd("data/clean/clean_20261014/center.bmp"); gref = [grad(c[y - H:y + H + 1, x - H:x + H + 1].astype(np.float32)) for x, y in PINS]
ts, tr, diff = [], [], []
for rep in range(3):
    for r in pick:
        t0 = time.perf_counter(); im = rd(r["path"]); t1 = time.perf_counter()
        f = np.array([float(v) for v in r["features66"].split(";")])[58:].reshape(4, 2); out = []
        for k, (x, y) in enumerate(PINS):
            ox, oy = int(round(f[k, 0])), int(round(f[k, 1])); a = im[y + oy - H:y + oy + H + 1, x + ox - H:x + ox + H + 1].astype(np.float32)
            W = np.eye(2, 3, dtype=np.float32); _, W = cv2.findTransformECC(gref[k], grad(a), W, cv2.MOTION_TRANSLATION, crit, full, 5); out.append((ox + W[0, 2], oy + W[1, 2]))
        t2 = time.perf_counter()
        if rep: ts.append((t2 - t1) * 1e3); tr.append((t1 - t0) * 1e3)
ts, tr = np.array(ts).reshape(2, -1).min(0), np.array(tr).reshape(2, -1).min(0)
print(f"나사 4개 기울기 영상 맞춤 (30장, 1스레드, Python OpenCV {cv2.__version__}): 중앙값 {np.median(ts):.1f} ms, p95 {np.percentile(ts, 95):.1f} ms, 최대 {ts.max():.1f} ms")
print(f"BMP 읽기+회색 변환: 중앙값 {np.median(tr):.1f} ms")

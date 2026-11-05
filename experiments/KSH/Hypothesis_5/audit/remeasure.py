# 측정 독립 재현 : 같은 절차를 Python OpenCV 로 다시 구현해 C++ 측정값과 대조 (무작위 12장)
import cv2, csv, random, sys, numpy as np
D, IMG = sys.argv[1], sys.argv[2]
M = {r['stem']: r for r in csv.DictReader(open(f'{D}/roi_meas_set1.csv', encoding='utf-8-sig'))}
ref = cv2.imread(f'{IMG}/center.png', 0).astype(np.float32)
def pc(a, b):
    w = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    return cv2.phaseCorrelate(a.copy(), b.copy(), w)[0]
def ecc(t, im, init):
    w = np.array([[1, 0, init[0]], [0, 1, init[1]]], np.float32)
    cv2.findTransformECC(t, im, w, cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7), None, 5)
    return w[0, 2], w[1, 2]
def region(cur, cx, cy, hw, g, wide):
    if wide:
        W = 2 * hw; gi = np.round(g).astype(int)
        a = ref[cy - W:cy + W, cx - W:cx + W]; b = cur[cy - W + gi[1]:cy + W + gi[1], cx - W + gi[0]:cx + W + gi[0]]
        d = pc(a, b); g = (gi[0] + d[0], gi[1] + d[1])
    gi = (int(round(g[0])), int(round(g[1])))
    t = ref[cy - hw:cy + hw, cx - hw:cx + hw]; c = cur[cy - hw + gi[1]:cy + hw + gi[1], cx - hw + gi[0]:cx + hw + gi[0]]
    dx, dy = ecc(t, c, (g[0] - gi[0], g[1] - gi[1]))
    return dx + gi[0]
random.seed(3); stems = random.sample(sorted(M), 12); mx = 0
for st in stems:
    cur = cv2.imread(f'{IMG}/{st}.png', 0).astype(np.float32)
    c = (1901, 1068); a = ref[c[1] - 200:c[1] + 200, c[0] - 450:c[0] + 450]; b = cur[c[1] - 200:c[1] + 200, c[0] - 450:c[0] + 450]
    g = pc(a, b)
    s = np.mean([region(cur, x, 1065, 110, g, False) for x in (1650, 1907, 2165)])
    w2 = region(cur, 2867, 1357, 60, g, True); w3 = region(cur, 1010, 1677, 60, g, True)
    e = [s - float(M[st]['s']), w2 - float(M[st]['wall_W2_dx']), w3 - float(M[st]['wall_W3_dx'])]
    mx = max(mx, *map(abs, e)); print(f'{st:8s} s {s:9.4f} (C++ {float(M[st]["s"]):9.4f})  W2 diff {e[1]:+.5f}  W3 diff {e[2]:+.5f}')
print('max abs diff px', mx)

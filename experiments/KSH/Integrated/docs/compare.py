# 두 분석 비교 : 같은 사진·같은 규칙 (교차 세션 / 2차 좌표 보류)
import sys, os, csv, re, glob, json, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/roi2')
os.chdir('/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/roi2')
import doe
from doe import PHOTOS, TRAIN, SET2, DEV, NEAR, TEST, CENTERS, roi_dx, fit, inv, score, summarize
H = '/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/merge'
OUT = f'{H}/results'; os.makedirs(OUT, exist_ok=True)
TICK_Y, TICK_T = 0.05, 0.01

# ---------------- 팀원 특징 (Fourier 코너) → QR 별 평균 이동
F = {}
for fn in glob.glob(f'{H}/shin/build/feat_0*.csv'):
    for r in csv.DictReader(open(fn)):
        if r['qr'] == '-1': F.setdefault(r['photo'], {}); continue
        if r['accepted'] != '1': continue
        F.setdefault(r['photo'], {})[(r['payload'], int(r['id']))] = (float(r['x']), float(r['y']))
MS = {}
for fn in glob.glob(f'{H}/shin/build/feat_0*.csv'):
    for r in csv.DictReader(open(fn)):
        if r['ms']: MS[r['photo']] = float(r['ms'])
REF0 = next(p for p in F if p.endswith('/img/center.png')); REF1 = CENTERS['center1']
def corner_qr6(p):
    ref = REF0 if PHOTOS[p]['set'] == 1 else REF1
    a, b = F.get(p), F.get(ref)
    if not a or not b: return None
    out = []
    for pay in ['FP11L', 'FP11C', 'FP11R']:
        ks = [k for k in a if k[0] == pay and k in b]
        if len(ks) < 8: return None
        d = np.array([[a[k][0] - b[k][0], a[k][1] - b[k][1]] for k in ks])
        out += [d[:, 0].mean(), d[:, 1].mean()]
    return np.array(out)   # Ldx, Ldy, Cdx, Cdy, Rdx, Rdy

# ---------------- 내 측정 (IC-GN)
QR = ['QR_L', 'QR_C', 'QR_R']; HOLES = ['W2_k30', 'W3_k30']
STRIPS_DY = ['upL_pure', 'upR_pure', 'lowL_pure', 'lowR_pure', 'doorL_pure', 'doorR_pure']
def refkey(p): return 'ref0' if PHOTOS[p]['set'] == 1 else 'ref1'
def dxdy(p, rois):
    m = doe.D.get((refkey(p), p))
    if m is None: return None
    v = []
    for r in rois:
        dx, dy, ok = m.get(r, (np.nan, np.nan, False))
        if not ok: return None
        v.append((dx, dy))
    return np.array(v)

def features(p, kind):
    """kind → 입력 벡터. sd* 는 물리 2특징 (s, d), 나머지는 역회귀 입력"""
    q = dxdy(p, QR); h = dxdy(p, HOLES)
    if kind == 'sd_icgn':
        if q is None or h is None: return None
        s = q[:, 0].mean(); return np.array([s, s - h[:, 0].mean()])
    if kind == 'sd_corner':
        c = corner_qr6(p)
        if c is None or h is None: return None
        # 코너 이동은 기준이 같은 영상(center/center1) — IC-GN 과 같은 기준
        s = c[[0, 2, 4]].mean(); return np.array([s, s - h[:, 0].mean()])
    if kind == 'qr6_corner':
        return corner_qr6(p)
    if kind == 'qr6_icgn':
        return None if q is None else q.reshape(-1)
    if kind == 'qr6_icgn+strip_dy':
        st = dxdy(p, STRIPS_DY)
        return None if q is None or st is None else np.r_[q.reshape(-1), st[:, 1]]
    if kind == 'qr6_icgn+holes':
        return None if q is None or h is None else np.r_[q.reshape(-1), h.reshape(-1)]
    if kind == 'sd+strip_dy':
        st = dxdy(p, STRIPS_DY)
        if q is None or h is None or st is None: return None
        s = q[:, 0].mean(); return np.r_[s, s - h[:, 0].mean(), st[:, 1]]
    raise ValueError(kind)

MODELS = [  # (이름, 특징, 방식, 출처)
    ('ksh 물리식 (IC-GN QR + 홀 시차)', 'sd_icgn', 'physical', 'ksh'),
    ('ksh 물리식 + Fourier 코너 QR', 'sd_corner', 'physical', '혼합'),
    ('shin QR6 역회귀 (Fourier 코너)', 'qr6_corner', 'inverse', 'shin'),
    ('QR6 역회귀 (IC-GN)', 'qr6_icgn', 'inverse', '대조'),
    ('QR6 + 홀 역회귀', 'qr6_icgn+holes', 'inverse', '혼합'),
    ('QR6 + 띠 dy 역회귀 (shin 가장자리 발상)', 'qr6_icgn+strip_dy', 'inverse', '혼합'),
    ('ksh 시차 + 띠 dy 역회귀', 'sd+strip_dy', 'inverse', '혼합'),
]

def fit_inverse(X, Y, ridge=1e-9):
    A = np.c_[np.ones(len(X)), X]; mu = A.mean(0); sd = A.std(0); sd[0] = 1; mu[0] = 0
    Z = (A - mu) / sd
    W = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ Y)
    return (W, mu, sd)
def pred_inverse(m, x):
    W, mu, sd = m; z = (np.r_[1.0, x] - mu) / sd; return z @ W

def train_predict(kind, how, train, test):
    tr = [(p, features(p, kind)) for p in train]; tr = [(p, x) for p, x in tr if x is not None]
    res = {}
    if how == 'physical':
        m = fit([(PHOTOS[p]['y'], PHOTOS[p]['t'], x[0], x[1]) for p, x in tr], 1, 1)
        for p in test:
            x = features(p, kind); res[p] = None if x is None else inv(m, x[0], x[1])
    else:
        m = fit_inverse(np.array([x for p, x in tr]), np.array([[PHOTOS[p]['y'], PHOTOS[p]['t']] for p, x in tr]))
        for p in test:
            x = features(p, kind); res[p] = None if x is None else tuple(pred_inverse(m, x))
    return res

def metrics(pred, paths):
    rows = [score(PHOTOS[p]['y'], PHOTOS[p]['t'], pred.get(p)) for p in paths]
    s = summarize(rows)
    a = np.array(rows, float); ok = np.isfinite(a[:, 1])
    s['tick_both'] = int(np.sum(ok & (np.abs(a[:, 1]) <= TICK_Y * 1000) & (np.abs(a[:, 2]) <= TICK_T)))
    s['spec_pass'] = int(np.sum(ok & (np.abs(a[:, 1]) <= 90) & (np.abs(a[:, 2]) <= 0.01)))
    return s

out = []
# P1 교차 세션 : 1차 단독축 240장 학습 → 2차·3rd
for name, kind, how, src in MODELS:
    pr = train_predict(kind, how, TRAIN, SET2 + TEST)
    for sub, paths in [('2차 목표 근처 33', NEAR), ('2차 학습 범위 안 45', DEV), ('3rd 234', TEST)]:
        m = metrics(pr, paths); out.append(dict(protocol='P1 교차 세션 (1차 학습)', model=name, source=src, subset=sub, **{k: v for k, v in m.items()}))
# P2 2차 좌표 보류 (팀원 규칙) : 19 좌표 중 하나씩 3장 통째로 보류
coords = sorted({(PHOTOS[p]['y'], PHOTOS[p]['t']) for p in SET2})
for name, kind, how, src in MODELS:
    pr = {}
    for c in coords:
        te = [p for p in SET2 if (PHOTOS[p]['y'], PHOTOS[p]['t']) == c]; tr = [p for p in SET2 if p not in te]
        pr.update(train_predict(kind, how, tr, te))
    for sub, paths in [('2차 57 전체', SET2), ('2차 Θ≤1 45', DEV)]:
        m = metrics(pr, paths); out.append(dict(protocol='P2 2차 좌표 보류 (shin 규칙)', model=name, source=src, subset=sub, **m))
# 팀원이 저장한 예측 (같은 57장)
def shin_csv(fn, prefix):
    rows = list(csv.DictReader(open(fn)))
    by = {}
    for r in rows:
        mm = re.search(r'Y\+([\d.]+) T\+([\d.]+) \((\d)\)', r['path'])
        key = (round(float(mm[1]), 3), round(float(mm[2]), 3), int(mm[3]))
        p = next(p for p in SET2 if (round(PHOTOS[p]['y'], 3), round(PHOTOS[p]['t'], 3), PHOTOS[p]['shot']) == key)
        by.setdefault(r['model'], {})[p] = (float(r['predicted_y_mm']), float(r['predicted_yaw_deg']))
    for model, pr in by.items():
        for sub, paths in [('2차 57 전체', SET2), ('2차 Θ≤1 45', DEV)]:
            m = metrics(pr, paths); out.append(dict(protocol='P2 2차 좌표 보류 (shin 규칙)', model=f'shin 저장값 · {prefix}{model}', source='shin', subset=sub, **m))
shin_csv(f'{H}/shin/docs/offaxis-eda/common-background-edges/coordinate_holdout.csv', '배경 가장자리 ')
shin_csv(f'{H}/shin/docs/offaxis-eda/plate-edges/coordinate_holdout.csv', '판 가장자리 ')

keys = ['protocol', 'model', 'source', 'subset', 'n', 'J', 'pass_', 'spec_pass', 'tick_both', 'y_rms', 't_rms', 'y_bias', 't_bias', 'fail']
with open(f'{OUT}/comparison_all.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, keys, extrasaction='ignore'); w.writeheader(); w.writerows(out)
for r in out:
    print(f"{r['protocol'][:2]} {r['subset']:14s} {r['model'][:42]:42s} J {r['J']:7.3f} pass {r['pass_']:3d} spec {r['spec_pass']:3d} tick {r['tick_both']:3d}/{r['n']:3d} Yrms {r['y_rms']:7.1f} Trms {r['t_rms']:.4f}")
json.dump(dict(ms_mean_shin=float(np.mean(list(MS.values()))), ms_median_shin=float(np.median(list(MS.values()))), n=len(MS)), open(f'{OUT}/timing_shin.json', 'w'))

"""재학습 실험 공용 함수 — C++ pose_model.cpp 의 fit_one / invert_pose_model 과 같은 계산"""
import json, numpy as np, math
F = json.load(open('/home/claude/rt/work/features.json'))
BAD = {('S4', 'Y-0.8 T-0.96')}          # 라벨 이상 (추정 Θ +0.72°) — 학습·평가 제외
REFN = ['center', 'center1', 'center2', 'center3']
SAMEDAY = {'S1': 0, 'S2': 1, 'S3': 1, 'S4': 0, 'C0': 0, 'C2': 1}   # 기존 관례 : 같은 날 기준 (S4 는 기준 없음 → center)

def quad(r):
    Y, T = r['Y'], r['T']
    if Y == 0 or T == 0: return 'AX'
    return 'Q1' if Y > 0 and T > 0 else 'Q2' if Y < 0 and T > 0 else 'Q3' if Y < 0 else 'Q4'

def samples(sets=('S1', 'S2', 'S3', 'S4'), ref='sameday', drop_bad=True):
    """ref : 'sameday' | 0..3 (기준 사진 번호) | 'mean' (기준 4장 평균)"""
    out = []
    for r in F:
        if r['set'] not in sets: continue
        if drop_bad and (r['set'], r['file']) in BAD: continue
        if ref == 'sameday': k = SAMEDAY[r['set']]; s, d = r[f's{k}'], r[f'd{k}']
        elif ref == 'mean': s = np.mean([r[f's{k}'] for k in range(4)]); d = np.mean([r[f'd{k}'] for k in range(4)])
        else: s, d = r[f's{ref}'], r[f'd{ref}']
        out.append(dict(file=r['file'], set=r['set'], Y=r['Y'], T=r['T'], s=s, d=d, q=quad(r)))
    return out

def basis(Y, T, order):
    return np.c_[[np.ones_like(Y), Y] + [T ** k for k in range(1, order + 1)]].T if False else np.column_stack([np.ones_like(Y), Y] + [T ** k for k in range(1, order + 1)])

def fit_one(S, key, order, w=None, c_fixed=None):
    Y = np.array([r['Y'] for r in S]); T = np.array([r['T'] for r in S]); v = np.array([r[key] for r in S])
    A = basis(Y, T, order)
    if c_fixed is not None: A2 = A[:, 1:]; vv = v - c_fixed
    else: A2 = A; vv = v
    if w is not None: sw = np.sqrt(np.asarray(w, float)); x, *_ = np.linalg.lstsq(A2 * sw[:, None], vv * sw, rcond=None)
    else: x, *_ = np.linalg.lstsq(A2, vv, rcond=None)
    if c_fixed is not None: x = np.r_[c_fixed, x]
    r = A @ x - v
    return dict(c=float(x[0]), a=float(x[1]), b=[float(t) for t in x[2:]], sigma=max(1e-6, float(np.sqrt(np.mean(r ** 2)))))

def fit(S, os_=1, od=1, w=None, c_fixed=None):
    cs = cd = None
    if c_fixed is not None: cs, cd = c_fixed
    return dict(s=fit_one(S, 's', os_, w, cs), d=fit_one(S, 'd', od, w, cd), os=os_, od=od)

def ev(m, Y, T):
    v = m['c'] + m['a'] * Y; dt = 0.0; tp = 1.0
    for k, b in enumerate(m['b']):
        dt += (k + 1) * b * tp; tp *= T; v += b * tp
    return v, m['a'], dt

def invert(M, s, d):
    S, D = M['s'], M['d']
    det = S['a'] * D['b'][0] - D['a'] * S['b'][0]
    rs, rd = s - S['c'], d - D['c']
    y = (rs * D['b'][0] - rd * S['b'][0]) / det; t = (S['a'] * rd - D['a'] * rs) / det
    ws, wd = 1 / S['sigma'] ** 2, 1 / D['sigma'] ** 2
    for _ in range(30):
        vs, sy, st = ev(S, y, t); vd, dy, dt = ev(D, y, t); es, ed = s - vs, d - vd
        a00 = ws * sy * sy + wd * dy * dy; a01 = ws * sy * st + wd * dy * dt; a11 = ws * st * st + wd * dt * dt
        b0 = ws * sy * es + wd * dy * ed; b1 = ws * st * es + wd * dt * ed; dd = a00 * a11 - a01 * a01
        ddy = (a11 * b0 - a01 * b1) / dd; ddt = (a00 * b1 - a01 * b0) / dd; y += ddy; t += ddt
        if abs(ddy) < 1e-10 and abs(ddt) < 1e-11: break
    return y, t

def predict(M, S):
    for r in S:
        r['ye'], r['te'] = invert(M, r['s'], r['d']); r['ey'] = (r['ye'] - r['Y']) * 1000; r['et'] = r['te'] - r['T']
    return S

def metrics(S):
    if not S: return None
    ey = np.array([r['ey'] for r in S]); et = np.array([r['et'] for r in S]); n = len(S)
    spec = (np.abs(ey) <= 90) & (np.abs(et) <= 0.01); goal = (np.abs(ey) <= 45) & (np.abs(et) <= 0.01); half = (np.abs(ey) <= 45) & (np.abs(et) <= 0.005)
    return dict(n=n, ym=ey.mean(), ysd=ey.std(), yrms=float(np.sqrt((ey ** 2).mean())), ymax=float(np.abs(ey).max()),
                tm=et.mean(), tsd=et.std(), trms=float(np.sqrt((et ** 2).mean())), tmax=float(np.abs(et).max()),
                spec=int(spec.sum()), goal=int(goal.sum()), half=int(half.sum()), y90=int((np.abs(ey) <= 90).sum()), t01=int((np.abs(et) <= 0.01).sum()))

def load_yml_model(path):
    import re
    t = open(path, encoding='utf-8-sig').read()
    def blk(name):
        b = t.split(name + ':')[1]
        c = float(re.search(r'c_px:\s*([-\d.e+]+)', b)[1]); a = float(re.search(r'a_px_per_mm:\s*([-\d.e+]+)', b)[1])
        ks = [float(x) for x in re.search(r'theta_px_per_deg_k:\s*\[([^\]]*)\]', b)[1].split(',')]
        sg = float(re.search(r'sigma_px:\s*([-\d.e+]+)', b)[1])
        return dict(c=c, a=a, b=ks, sigma=sg)
    return dict(s=blk('\n   s'), d=blk('\n   d'))

def write_yml_model(M, path, note, n):
    def f(m): return ('      c_px: %.16e\n      a_px_per_mm: %.16e\n      theta_px_per_deg_k: [ %s ]\n      sigma_px: %.16e\n'
                      % (m['c'], m['a'], ', '.join('%.16e' % b for b in m['b']), m['sigma']))
    s = ('%%YAML:1.0\n---\ntrained_at: "2026-11-10"\nnote: "%s"\nn_samples: %d\ntheta_order_s: %d\ntheta_order_d: %d\nmodel:\n   s:\n%s   d:\n%s'
         % (note, n, len(M['s']['b']), len(M['d']['b']), f(M['s']), f(M['d'])))
    open(path, 'w', newline='\r\n', encoding='utf-8').write(s)

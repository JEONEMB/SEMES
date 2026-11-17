# -*- coding: utf-8 -*-
"""
pose_model_tool.py — Y·Θ 보정식(s·d 다항식) 학습 · 평가 · 영점 보정 도구 (C++ pose_model.cpp 와 같은 식·같은 역산)

보정식 :  s = c_s + a_s·Y + Σ b_s,k·Θ^k        d = c_d + a_d·Y + Σ b_d,k·Θ^k
측정한 (s, d) 를 동시에 만족하는 (Y, Θ) 를 가우스–뉴턴으로 구함 → 모델 파일 형식은 config/model_*.yml 과 같음

명령
  train  : 특징 CSV → 모델 yml
      python pose_model_tool.py train --features F.csv --train S1,S4 --s-col s1 --d-col d1 \
             --zero-set S1 --zero-s-col s0 --zero-d-col d0 --order-s 1 --order-d 2 --out model.yml
      · 기울기(a, b)는 학습 세트 전체에서 '세트별 상수항'을 따로 두고 함께 맞춤 (촬영 날마다 다른 기준 차이를 흡수)
      · 세트마다 장수가 달라도 한 세트가 식을 끌고 가지 않도록 세트 균형 가중(1/장수) 사용 (--no-balance 로 끔)
      · 상수항 c 는 '같은 날 찍은 센터 사진' 이 있는 세트(--zero-set)에서, 그 센터 기준 특징(--zero-*-col)으로 구함
        = 센터 사진이 회귀선의 영점에서 얼마나 떨어져 찍혔는지 (로봇 백래시 · 접근 방향) 를 담는 값
  eval   : 모델 yml + 특징 CSV → 사진별 오차 · 요약 (명세 Y ±90 µm · Θ ±0.01°)
      python pose_model_tool.py eval --model model.yml --features F.csv --sets S2,S3 --s-col s1 --d-col d1 --out-csv pred.csv
  zero   : 장비에서 알고 있는 위치로 움직여 잰 결과(참값·추정값 CSV)로 상수 치우침만 없앤 모델을 씀
      python pose_model_tool.py zero --model model.yml --table zero.csv --out model_zero.yml
      zero.csv 열 : true_y_mm,true_t_deg,est_y_mm,est_t_deg  (부호 반대인 큰 이상값은 --max-abs 로 제외)

특징 CSV 열 : set, Y, T, 그리고 --s-col / --d-col 로 지정한 열 (C++ featdump / align_train --features 출력, 또는 text_roi 의 절대 특징)
"""
import argparse, csv, math, re, sys
import numpy as np

# ---------------------------------------------------------------- 식 · 역산 (pose_model.cpp 와 같은 계산)
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

# ---------------------------------------------------------------- 모델 파일
def load_model(path):
    t = open(path, encoding='utf-8-sig').read()
    def blk(name):
        b = t.split(name + ':')[1]
        g = lambda k: float(re.search(k + r':\s*([-\d.eE+]+)', b)[1])
        ks = [float(x) for x in re.search(r'theta_px_per_deg_k:\s*\[([^\]]*)\]', b)[1].split(',')]
        return dict(c=g('c_px'), a=g('a_px_per_mm'), b=ks, sigma=g('sigma_px'))
    return dict(s=blk('\n   s'), d=blk('\n   d'))

def save_model(M, path, note, n, date):
    def f(m): return ('      c_px: %.16e\n      a_px_per_mm: %.16e\n      theta_px_per_deg_k: [ %s ]\n      sigma_px: %.16e\n'
                      % (m['c'], m['a'], ', '.join('%.16e' % b for b in m['b']), m['sigma']))
    s = ('%%YAML:1.0\n---\ntrained_at: "%s"\nnote: "%s"\nn_samples: %d\ntheta_order_s: %d\ntheta_order_d: %d\nmodel:\n   s:\n%s   d:\n%s'
         % (date, note, n, len(M['s']['b']), len(M['d']['b']), f(M['s']), f(M['d'])))
    open(path, 'w', newline='\r\n', encoding='utf-8').write(s)

def read_csv(path):
    with open(path, encoding='utf-8-sig') as fh: return list(csv.DictReader(fh))

# ---------------------------------------------------------------- 학습
def fit_fixed_effects(groups, col, order, balance=True):
    """groups : [(Y, T, v), ...] 세트별. 기울기 공통 + 세트별 상수항. 반환 (a, b[], 세트별 c[], 잔차 rms)"""
    k = len(groups); rows = []; vals = []; w = []
    for i, (Y, T, v) in enumerate(groups):
        oh = np.zeros((len(Y), k)); oh[:, i] = 1
        rows.append(np.column_stack([oh, Y] + [T ** j for j in range(1, order + 1)])); vals.append(v)
        w.append(np.full(len(Y), 1.0 / len(Y) if balance else 1.0))
    A = np.vstack(rows); v = np.concatenate(vals); sw = np.sqrt(np.concatenate(w))
    x, *_ = np.linalg.lstsq(A * sw[:, None], v * sw, rcond=None)
    r = A @ x - v
    return float(x[k]), [float(t) for t in x[k + 1:]], [float(t) for t in x[:k]], float(np.sqrt(np.mean(r ** 2)))

def cmd_train(a):
    R = read_csv(a.features); sets = a.train.split(',')
    def arr(rows, c): return np.array([float(r[c]) for r in rows])
    M = {}; n = 0
    for key, col, zcol, order in [('s', a.s_col, a.zero_s_col, a.order_s), ('d', a.d_col, a.zero_d_col, a.order_d)]:
        groups = []
        for st in sets:
            rr = [r for r in R if r['set'] == st and r.get('exclude', '0') != '1']
            groups.append((arr(rr, 'Y'), arr(rr, 'T'), arr(rr, col)))
        n = sum(len(g[0]) for g in groups)
        sl, bs, cs, rms = fit_fixed_effects(groups, col, order, not a.no_balance)
        # 상수항 : 같은 날 센터가 있는 세트에서 그 센터 기준 특징으로 (기울기 고정, 잔차 평균)
        zr = [r for r in R if r['set'] == a.zero_set and r.get('exclude', '0') != '1']
        Y, T, v = arr(zr, 'Y'), arr(zr, 'T'), arr(zr, zcol or col)
        c = float(np.mean(v - (sl * Y + sum(b * T ** (j + 1) for j, b in enumerate(bs)))))
        M[key] = dict(c=c, a=sl, b=bs, sigma=max(1e-6, rms))
        print(f'  {key} = {c:+.6f} {sl:+.6f}·Y ' + ' '.join(f'{b:+.6f}·Θ^{j+1}' for j, b in enumerate(bs)) + f'   (σ {rms:.4f} px, 세트별 상수 {np.round(cs,4).tolist()})')
    save_model(M, a.out, a.note or f'train {a.train} (set fixed effects, balanced={not a.no_balance}), c from {a.zero_set}', n, a.date)
    print('저장 :', a.out, f'({n}장)')

# ---------------------------------------------------------------- 평가
def summarize(ey, et, tag=''):
    ey = np.asarray(ey); et = np.asarray(et); n = len(ey)
    spec = (np.abs(ey) <= 90) & (np.abs(et) <= 0.01); half = (np.abs(ey) <= 45) & (np.abs(et) <= 0.005)
    print(f'{tag} n={n}  Y 평균 {ey.mean():+.1f} µm  σ {ey.std():.1f}  rms {math.sqrt((ey**2).mean()):.1f}  최대 {np.abs(ey).max():.1f} | '
          f'Θ 평균 {et.mean():+.4f}°  rms {math.sqrt((et**2).mean()):.4f}  최대 {np.abs(et).max():.4f} | 명세 {spec.sum()}/{n}  절반기준 {half.sum()}/{n}')

def cmd_eval(a):
    M = load_model(a.model); R = read_csv(a.features); sets = a.sets.split(',')
    out = []
    for r in R:
        if r['set'] not in sets or r.get('exclude', '0') == '1': continue
        y, t = invert(M, float(r[a.s_col]), float(r[a.d_col]))
        out.append(dict(set=r['set'], file=r.get('file', ''), Y=float(r['Y']), T=float(r['T']), y_est=y, t_est=t,
                        ey_um=(y - float(r['Y'])) * 1000, et_deg=t - float(r['T'])))
    for st in sets:
        o = [x for x in out if x['set'] == st]
        if o: summarize([x['ey_um'] for x in o], [x['et_deg'] for x in o], f'[{st}]')
    summarize([x['ey_um'] for x in out], [x['et_deg'] for x in out], '[전체]')
    if a.out_csv:
        with open(a.out_csv, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

# ---------------------------------------------------------------- 영점(상수 치우침) 보정
def shift_model(m, by, bt):
    """m(y + by, t + bt) 를 같은 형식의 다항식으로 전개 → 새 모델은 같은 특징에서 (by, bt) 만큼 작게 읽음"""
    b = m['b']; K = len(b)
    c = m['c'] + m['a'] * by + sum(b[k] * bt ** (k + 1) for k in range(K))
    nb = [0.0] * K
    for k in range(K):                       # (t+bt)^(k+1) = Σ_j C(k+1, j) t^j bt^(k+1-j)
        p = k + 1
        for j in range(1, p + 1): nb[j - 1] += b[k] * math.comb(p, j) * bt ** (p - j)
    return dict(c=c, a=m['a'], b=nb, sigma=m['sigma'])

def cmd_zero(a):
    M = load_model(a.model); R = read_csv(a.table)
    ey = np.array([float(r['est_y_mm']) - float(r['true_y_mm']) for r in R])
    et = np.array([float(r['est_t_deg']) - float(r['true_t_deg']) for r in R])
    k = (np.abs(et) <= a.max_abs_t) & (np.abs(ey) <= a.max_abs_y)
    by, bt = float(np.median(ey[k])) if a.median else float(ey[k].mean()), float(np.median(et[k])) if a.median else float(et[k].mean())
    print(f'사용 {k.sum()}/{len(k)}행 (제외 {np.where(~k)[0] + 1})  치우침 Y {by * 1000:+.1f} µm, Θ {bt:+.4f}°')
    N = dict(s=shift_model(M['s'], by, bt), d=shift_model(M['d'], by, bt))
    save_model(N, a.out, f'zero-calibrated from {a.table}: bias Y {by*1000:+.1f} um, T {bt:+.4f} deg (base {a.model})', int(k.sum()), a.date)
    print('저장 :', a.out)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest='cmd', required=True)
    t = sp.add_parser('train'); t.add_argument('--features', required=True); t.add_argument('--train', required=True)
    t.add_argument('--s-col', required=True); t.add_argument('--d-col', required=True); t.add_argument('--zero-set', required=True)
    t.add_argument('--zero-s-col'); t.add_argument('--zero-d-col'); t.add_argument('--order-s', type=int, default=1); t.add_argument('--order-d', type=int, default=2)
    t.add_argument('--no-balance', action='store_true'); t.add_argument('--out', required=True); t.add_argument('--note', default=''); t.add_argument('--date', default='')
    e = sp.add_parser('eval'); e.add_argument('--model', required=True); e.add_argument('--features', required=True); e.add_argument('--sets', required=True)
    e.add_argument('--s-col', required=True); e.add_argument('--d-col', required=True); e.add_argument('--out-csv')
    z = sp.add_parser('zero'); z.add_argument('--model', required=True); z.add_argument('--table', required=True); z.add_argument('--out', required=True)
    z.add_argument('--max-abs-y', type=float, default=0.5); z.add_argument('--max-abs-t', type=float, default=0.1); z.add_argument('--median', action='store_true'); z.add_argument('--date', default='')
    a = p.parse_args(); {'train': cmd_train, 'eval': cmd_eval, 'zero': cmd_zero}[a.cmd](a)

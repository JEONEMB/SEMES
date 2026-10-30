"""Y 사진을 Θ로 헷갈리는 원인 진단.
(a) 숨은 Θ: Y 사진 안에 실제 Θ 움직임이 섞였나. 판밖 점을 '측정 도구'로만 써서 (Y, Θ) 동시 추정 (판정에는 안 씀). 짝/홀 교차검증.
(b) 그 숨은 Θ가 QR만 판정에서 틀린 Y 사진과 관계 있나.
(c) 같은 데이터, 같은 점에서 매핑(1차 / 2차)만 바꾸면 틀리는 방향이 바뀌나."""
import numpy as np, pandas as pd
from common import *
from t6_F1 import V5, nuis
from t7_F4 import FOLD
from t12_qr_analysis import classify

Z = np.load(HERE/'tracks_hp8.npz'); F = pd.read_csv(HERE/'features_hp8.csv'); names = list(Z['names']); D = Z['D_gn']
ax = np.array([AXIS[n] for n in names]); cmd = np.array([VAL[n] for n in names])
Dm = D.reshape(len(names), -1); est = np.full((len(names), 2), np.nan)
for k in (0, 1):
    te = np.array([FOLD.get(n, -1) == k for n in names]); tr = ~te
    lab = np.array([V5.CORR[k][V5.NAMES.index(n)] for n in names])
    S = {}
    for a in ['Y', 'Θ']:
        m = tr & ((ax == a) | (ax == 'C')); A = np.c_[lab[m], np.ones(m.sum())]; co, *_ = np.linalg.lstsq(A, Dm[m], rcond=None); S[a] = co
    var = np.var(Dm[tr & (ax == 'Y')] - np.c_[lab[tr & (ax == 'Y')], np.ones((tr & (ax == 'Y')).sum())] @ S['Y'], 0)
    w = np.sqrt(1 / np.maximum(var, 0.003**2)); b = (S['Y'][1] + S['Θ'][1]) / 2
    U = nuis(F.x.values.astype(float), F.y.values.astype(float), 1); A = np.c_[S['Y'][0], S['Θ'][0], U]
    for i in np.where(te)[0]:
        p, *_ = np.linalg.lstsq(A * w[:, None], (Dm[i] - b) * w, rcond=None); est[i] = p[:2]
y = ax == 'Y'; th = ax == 'Θ'
hidT = est[y, 1]; hidY = est[th, 0]
print(f'(a) Y 사진 속 숨은 Θ: 평균 {hidT.mean() * 1000:+.2f} mΘ, 표준편차 {hidT.std() * 1000:.2f} mΘ, 최대 {np.abs(hidT).max() * 1000:.2f} mΘ'
      f'  (같은 QR 이동의 Y로 환산 표준편차 {hidT.std() * 80.66 / 14.39 * 1000:.1f}µm)')
print(f'    Θ 사진 속 숨은 Y: 표준편차 {hidY.std() * 1000:.1f}µm')
ses = np.array([session_of(n) for n in names])
for s in ['Y1+', 'Y1-', 'Y2+', 'Y2-']:
    m = ses[y] == s; print(f'    {s}: 숨은 Θ 평균 {hidT[m].mean() * 1000:+.2f} mΘ, 표준편차 {hidT[m].std() * 1000:.2f}')
# (b) QR만 판정 (17px, 2차, 보정본)과 비교
Fq = pd.read_csv(HERE/'features_qr_hp8.csv'); Zq = np.load(HERE/'tracks_qr_hp8.npz'); mq = Fq.region.str.startswith('QR').values
res = {}
for quad in (False, True):
    df = classify(Zq['D_gn'][:, mq], list(Zq['names']), Fq[mq].reset_index(drop=True), lambda D_, n_, F_: np.ones(len(F_), bool), quad=quad)
    res[quad] = df; bad = df[df.axis != df.pred_axis]
    print(f'(c) {"2차" if quad else "1차"} 매핑: 틀림 Y {((bad.axis == "Y")).sum()}장, Θ {((bad.axis == "Θ")).sum()}장 →', ', '.join(f'{r.axis}{r.cmd:+.2f}' for r in bad.sort_values("cmd").itertuples()))
df = res[True].set_index('file'); yy = [n for n in df.index if AXIS[n] == 'Y']
hid = pd.Series(est[:, 1], index=names)[yy]; wrong = df.loc[yy, 'pred_axis'] == 'Θ'; small = df.loc[yy, 'cmd'].abs() <= 0.3
print(f'(b) |Y| ≤ 0.3mm 사진 중 틀린 것의 숨은 Θ 절댓값 평균 {hid[wrong & small].abs().mean() * 1000:.2f} mΘ ({(wrong & small).sum()}장), '
      f'맞은 것 {hid[~wrong & small].abs().mean() * 1000:.2f} mΘ ({(~wrong & small).sum()}장)')
print('    틀린 Y 사진: ' + ', '.join(f'{n} 숨은Θ {hid[n] * 1000:+.2f}m 비용비 {df.loc[n, "cost_ratio"]:.2f}' for n in hid[wrong].index))
# 판정에 쓴 숨은 Θ가 QR 패턴으로 보면 얼마의 Y 오판에 해당하나: 숨은 Θ 1 mΘ ≈ Y 5.6µm 의 QR 이동
pd.DataFrame(dict(file=names, axis=ax, cmd=cmd, est_Y=est[:, 0], est_T=est[:, 1])).to_csv(HERE/'t13_hidden.csv', index=False, encoding='utf-8-sig')

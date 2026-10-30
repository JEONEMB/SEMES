"""4~7단계: 이동 무늬 분석, 매핑 함수, 맞추기(Y인지 Θ인지 판정 + 값), 8조건 평가.
조건 = 정답 라벨(raw / 보정본) x 영역(QR / QR+나사) x 측정법(PC / PC+GN) x 매핑 함수(1차 / 2차).

매핑 함수 (패치 p, 성분 x/y마다): d_p = s_p(축) * v [+ c_p(축) * v^2] + b_p(축). 축(Y, Θ)마다 그 축 사진으로 따로 맞춘다.
2차항은 회전 중심이 QR에서 떨어져 있을 때 생기는 휨 (v4의 ③).
맞추기: 새 사진 이동장 d에서 가설 h(Y 또는 Θ)마다
   min_{v, u}  Σ w (d − f_h(v) − U u)^2   (U = 방해 무늬: 세로 평행 이동, 화면 내 회전)
   2차 함수는 v에 대해 비선형이라 가우스-뉴턴으로 반복해서 푼다 (1차 해에서 출발).
   가중 w = 1 / 패치 잡음 분산 (학습 잔차), Huber로 튀는 패치를 덜 믿는다.
   비용이 작은 가설을 고르고 그 v를 값으로 낸다.
채점: 명령값 기준, 실제 위치 기준(보정본, 학습 폴드로만 맞춘 ①로 계산) 둘 다.
0.09 통과 규칙: 축을 맞혔고 |오차| ≤ 0.09. 축을 틀렸으면 실패 (단, 참값과 예측값이 둘 다 0.09 안이면 통과)."""
import os, io, contextlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from step1_2 import HERE, PROJ, X0, Y0, HH, WW, FOLD, q
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
TXT, SUB, GRID = '#1f1f1e', '#5f5e58', '#e4e3dc'
BLUE, ORG, GRN, RED = '#4a72d0', '#d2703f', '#5aa776', '#c8423b'

Z = np.load(HERE/'field.npz'); F = Z['F'].astype(np.float64); NAMES = list(Z['names'])   # F: 측정법, 기준, 사진, 패치, 2
P = pd.read_csv(HERE/'patches.csv'); P = P[P.use].reset_index(drop=True); NP = len(P)
AXIS = np.array(['C' if n == 'center.bmp' else ('Y' if n[0] == 'Y' else 'Θ') for n in NAMES])
VAL = np.array([0.0 if n == 'center.bmp' else q.loc[n, 'value'] for n in NAMES])
FOLDA = np.array([-1 if n == 'center.bmp' else FOLD[n] for n in NAMES])

# ---------- 보정본 라벨: v4의 ① 백래시 + 드리프트를 폴드마다 학습 사진으로만 맞춘다 ----------
V4 = PROJ/'analysis_v4/zero_align.py'
_g = {'__file__': str(V4)}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(V4, encoding='utf-8').read().split('rows, RES')[0], _g)
def corrected_labels(k):
    """폴드 k가 시험일 때의 보정본 라벨 (모든 사진). 실제 위치 = 명령값 + o/k1."""
    out = VAL.copy()
    for a in ['Y', 'Θ']:
        g = _g['AX'][a]; tr = np.array([FOLD[f] != k for f in g.file])
        o, k1, *_ = _g['m1_backlash_drift'](g, tr)
        for f, vv in zip(g.file, g.value.values + o / k1): out[NAMES.index(f)] = vv
    return out
CORR = {k: corrected_labels(k) for k in (0, 1)}

# ---------- 방해 무늬 ----------
xc, yc = P.cx.values - P.cx.mean(), P.cy.values - P.cy.mean()
U = np.stack([np.c_[np.zeros(NP), np.ones(NP)].ravel(),                 # 세로 평행 이동
              np.c_[-yc, xc].ravel() / 100], 1)                           # 화면 내 회전
def mask_rows(region):
    m = P.region.str.startswith('QR').values if region == 'QR' else np.ones(NP, bool)
    return np.repeat(m, 2)

def fit_map(D, v, rows, quad=False):
    """D: 사진 x (2NP) 이동장, v: 라벨. 패치 성분마다 1차 또는 2차. 반환 (s, c, b), 잔차 분산."""
    A = np.c_[v, v**2, np.ones_like(v)] if quad else np.c_[v, np.ones_like(v)]
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)
    res = D - A @ coef; var = res.var(0) * len(v) / max(len(v) - A.shape[1], 1)
    if not quad: coef = np.stack([coef[0], np.zeros_like(coef[0]), coef[1]])
    return coef, var

def solve(d, coef, w, rows, huber=3.0, iters=6):
    """가우스-뉴턴 + Huber 재가중: f(v) = s v + c v^2 + b 를 v0 근처에서 선형화해 (dv, u)를 푼다."""
    s, c, b = coef[0][rows], coef[1][rows], coef[2][rows]; dd = d[rows]; base = w[rows]; ww = base.copy(); v = 0.0
    for it in range(iters):
        J = s + 2 * c * v; z = dd - (s * v + c * v * v + b)
        A = np.c_[J, U[rows]]; sw = np.sqrt(ww)
        p, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None); v += p[0]
        r = (dd - (s * v + c * v * v + b) - U[rows] @ p[1:]) * np.sqrt(base)
        ww = base * np.where(np.abs(r) <= huber, 1, huber / np.maximum(np.abs(r), 1e-12))
    # 마지막 v에서 방해 무늬만 다시 맞추고 비용 계산
    z = dd - (s * v + c * v * v + b); sw = np.sqrt(ww)
    u, *_ = np.linalg.lstsq(U[rows] * sw[:, None], z * sw, rcond=None)
    rr = (z - U[rows] @ u) * np.sqrt(base); cost = np.sum(np.where(np.abs(rr) <= huber, rr**2, 2 * huber * np.abs(rr) - huber**2))
    return v, cost

def run_condition(label, region, meth, quad=False):
    rows = mask_rows(region); recs = []
    for k in (0, 1):
        D = F[meth, k].reshape(len(NAMES), -1)
        lab = VAL if label == 'raw' else CORR[k]
        tr = FOLDA != k
        maps = {}
        for a in ['Y', 'Θ']:
            m = tr & ((AXIS == a) | (AXIS == 'C'))
            maps[a] = fit_map(D[m], lab[m], rows, quad)
        var = np.maximum((maps['Y'][1] + maps['Θ'][1]) / 2, 0.003**2); w = 1 / var
        for i in np.where(FOLDA == k)[0]:
            out = {a: solve(D[i], maps[a][0], w, rows) for a in ['Y', 'Θ']}
            pa = min(out, key=lambda a: out[a][1])
            recs.append(dict(file=NAMES[i], axis=AXIS[i], cmd=VAL[i], act=CORR[k][i], pred_axis=pa, pred=out[pa][0],
                             cost_ratio=out['Y'][1] / out['Θ'][1], predY=out['Y'][0], predT=out['Θ'][0]))
    return pd.DataFrame(recs)

def score(df, basis):
    t = df[basis].values; ok_axis = (df.pred_axis == df.axis).values
    err = np.where(ok_axis, df.pred.values - t, np.nan)
    passed = np.where(ok_axis, np.abs(df.pred.values - t) <= 0.09, (np.abs(t) <= 0.09) & (np.abs(df.pred.values) <= 0.09))
    y, th = (df.axis == 'Y').values, (df.axis == 'Θ').values
    return dict(Y_RMS_um=np.sqrt(np.nanmean(err[y]**2)) * 1000, Y_max_um=np.nanmax(np.abs(err[y])) * 1000,
                T_RMS_m=np.sqrt(np.nanmean(err[th]**2)) * 1000, T_max_m=np.nanmax(np.abs(err[th])) * 1000,
                pass09=passed.mean() * 100)

if __name__ == '__main__':
    CONDS = [(lab, reg, me, qd) for lab in ['raw', '보정본'] for reg in ['QR', 'QR+나사'] for me in [0, 1] for qd in [False, True]]
    MN = {0: 'PC', 1: 'PC+GN'}
    rows, PRED = [], {}
    for lab, reg, me, qd in CONDS:
        df = run_condition(lab, reg, me, qd); key = f'{lab} / {reg} / {MN[me]} / {"2차" if qd else "1차"}'; PRED[key] = df
        wrongY = ((df.axis == 'Y') & (df.pred_axis == 'Θ')).sum(); wrongT = ((df.axis == 'Θ') & (df.pred_axis == 'Y')).sum()
        r = dict(조건=key, 라벨=lab, 영역=reg, 측정법=MN[me], 매핑=('2차' if qd else '1차'), 판정오류_Y사진=wrongY, 판정오류_Θ사진=wrongT)
        for b, nm in [('cmd', '명령값'), ('act', '실제위치')]:
            for kk, vv in score(df, b).items(): r[f'{nm}_{kk}'] = vv
        rows.append(r); print(key, wrongY, wrongT, {k: round(v, 2) for k, v in score(df, 'act').items()}, flush=True)
    T = pd.DataFrame(rows); T.round(3).to_csv(PROJ/'40_결과표_패치이동장_16조건.csv', index=False, encoding='utf-8-sig')
    pd.concat([d.assign(조건=k) for k, d in PRED.items()]).to_csv(HERE/'predictions.csv', index=False, encoding='utf-8-sig')

    # ---------- 그림 38: 합성 검증 ----------
    S = pd.read_csv(HERE/'synthetic.csv')
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, col, ttl in [(axs[0], '편향', '편향 (잰 값 - 참값, 패치 평균)'), (axs[1], '흩어짐', '흩어짐 (패치 사이 표준편차)')]:
        for (me, nz), g in S[S.방향 == 'dx'].groupby(['측정법', '잡음']):
            ax.plot(g.이동, g[col], 'o-' if nz == 0 else 's--', color=BLUE if me == 'PC' else ORG, label=f'{me}, 잡음 {nz:.0f}')
        ax.axhline(0, color=SUB, lw=0.8); ax.set_xlabel('일부러 옮긴 양 (px)'); ax.set_ylabel('px'); ax.grid(color=GRID)
        ax.set_title(ttl, loc='left', fontsize=13); ax.legend(frameon=False)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    fig.suptitle('3단계 합성 검증: 기준 영상을 알고 있는 양만큼 옮겨서 패치 이동을 다시 잼 (dx 성분, 잡음 = 밝기 표준편차 1)', x=0.01, ha='left', fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig(PROJ/'38_3단계_합성검증.png', dpi=110, facecolor='white'); plt.close()

    # ---------- 그림 39: 이동 무늬 지도 (보정본, PC+GN, 폴드0 학습) ----------
    D = F[1, 0].reshape(len(NAMES), -1); tr = FOLDA != 0; lab = CORR[0]; allr = np.ones(2 * NP, bool)
    mp = {a: fit_map(D[tr & ((AXIS == a) | (AXIS == 'C'))], lab[tr & ((AXIS == a) | (AXIS == 'C'))], allr) for a in ['Y', 'Θ']}
    mp = {a: (mp[a][0][0], None, mp[a][1]) for a in mp}       # (s, -, 잔차 분산)
    sY, sT = mp['Y'][0].reshape(NP, 2), mp['Θ'][0].reshape(NP, 2)
    w = 1 / np.maximum((mp['Y'][2] + mp['Θ'][2]) / 2, 0.003**2)
    # Y 1mm가 만드는 무늬를 Θ로 가장 비슷하게 흉내 냈을 때 남는 무늬 (+방해 무늬까지 허용) = 구분 단서
    def leftover(rows):
        A = np.c_[mp['Θ'][0][rows], U[rows]]; sw = np.sqrt(w[rows])
        p, *_ = np.linalg.lstsq(A * sw[:, None], mp['Y'][0][rows] * sw, rcond=None); r = mp['Y'][0][rows] - A @ p
        return r, p[0], np.sqrt(np.sum(w[rows] * r**2))
    rQ, aQ, nQ = leftover(mask_rows('QR')); rA, aA, nA = leftover(allr)
    fig, axs = plt.subplots(3, 1, figsize=(18, 15))
    for ax, vec, ttl, sc in [(axs[0], sY, f'Y 1mm가 만드는 이동 무늬 s_Y (평균 {np.abs(sY[:, 0]).mean():.2f}px)', None),
                             (axs[1], sT, f'Θ 1이 만드는 이동 무늬 s_Θ (평균 {np.abs(sT[:, 0]).mean():.2f}px)', None)]:
        R0 = np.load(HERE/'refs.npz')['mean_test0']; ax.imshow(R0, cmap='gray', extent=(X0, X0 + WW, Y0 + HH, Y0), alpha=0.55)
        dev = vec - vec.mean(0)   # 평균 이동을 빼고 패치 사이 차이만 확대
        Q = ax.quiver(P.cx, P.cy, dev[:, 0], -dev[:, 1], np.hypot(*dev.T), cmap='viridis', angles='xy', scale_units='xy', scale=1 / 400, width=0.0016)
        ax.set_title(ttl + f'. 화살표 = 평균 이동 ({vec.mean(0)[0]:+.2f}, {vec.mean(0)[1]:+.2f})px 을 뺀 나머지 x400', loc='left', fontsize=13)
        plt.colorbar(Q, ax=ax, fraction=0.015, label='px'); ax.set_xlim(X0, X0 + WW); ax.set_ylim(Y0 + HH, Y0)
    ax = axs[2]; ax.imshow(np.load(HERE/'refs.npz')['mean_test0'], cmap='gray', extent=(X0, X0 + WW, Y0 + HH, Y0), alpha=0.55)
    rv = rA.reshape(NP, 2)
    Q = ax.quiver(P.cx, P.cy, rv[:, 0], -rv[:, 1], np.hypot(*rv.T), cmap='magma', angles='xy', scale_units='xy', scale=1 / 400, width=0.0016)
    plt.colorbar(Q, ax=ax, fraction=0.015, label='px'); ax.set_xlim(X0, X0 + WW); ax.set_ylim(Y0 + HH, Y0)
    zq, za = 3 / nQ, 3 / nA
    ax.set_title(f'구분 단서: Y 1mm 무늬를 Θ {aA:+.4f} + 방해 무늬로 최대한 흉내 내고 남은 차이 x400.  '
                 f'잡음 대비 크기 QR만 {nQ:.1f}, QR+나사 {nA:.1f} → 3σ로 구분되는 최소 |Y| 약 QR만 {zq:.2f}mm, QR+나사 {za:.2f}mm',
                 loc='left', fontsize=12)
    plt.tight_layout(); plt.savefig(PROJ/'39_4단계_이동무늬지도.png', dpi=105, facecolor='white'); plt.close()
    print('구분 단서', nQ, nA, zq, za)

    # ---------- 그림 40: 8조건 결과 히트맵 ----------
    cols = [('판정오류_Y사진', '판정 오류\nY 사진', 'd'), ('판정오류_Θ사진', '판정 오류\nΘ 사진', 'd'),
            ('명령값_Y_RMS_um', 'Y RMS µm\n(명령값)', '.1f'), ('실제위치_Y_RMS_um', 'Y RMS µm\n(실제 위치)', '.1f'),
            ('명령값_T_RMS_m', 'Θ RMS mΘ\n(명령값)', '.2f'), ('실제위치_T_RMS_m', 'Θ RMS mΘ\n(실제 위치)', '.2f'),
            ('명령값_pass09', '0.09 통과 %\n(명령값)', '.1f'), ('실제위치_pass09', '0.09 통과 %\n(실제 위치)', '.1f')]
    fig, ax = plt.subplots(figsize=(18, 10))
    M = np.array([[T[c].values for c, _, _ in cols]]).squeeze().T
    Mn = np.zeros_like(M, float)
    for j, (c, _, _) in enumerate(cols):
        col = M[:, j].astype(float); lo, hi = col.min(), col.max(); z = (col - lo) / (hi - lo + 1e-12)
        Mn[:, j] = 1 - z if 'pass' in c else z
    ax.imshow(Mn, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j, (c, _, f) in enumerate(cols): ax.text(j, i, format(M[i, j], f) if f != 'd' else str(int(M[i, j])), ha='center', va='center', fontsize=12)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l, _ in cols], fontsize=11)
    ax.set_yticks(range(len(T))); ax.set_yticklabels(T.조건, fontsize=12)
    ax.set_title('5~7단계: 16조건 결과 (짝수/홀수 교차검증, 열마다 초록 = 좋음). RMS는 축을 맞힌 사진만, 축을 틀리면 0.09 통과 실패',
                 loc='left', fontsize=13)
    plt.tight_layout(); plt.savefig(PROJ/'40_결과_패치이동장_16조건.png', dpi=110, facecolor='white'); plt.close()

    # ---------- 그림 41: 조건별 오차 (실제 위치 기준) ----------
    fig, axs = plt.subplots(4, 8, figsize=(34, 17), sharey='row')
    for jj, (key, df) in enumerate(PRED.items()):
        j, blk = jj // 2, jj % 2                       # 같은 열 = 같은 조건의 1차(위 2행) / 2차(아래 2행)
        for i0, a in enumerate(['Y', 'Θ']):
            i = 2 * blk + i0; ax = axs[i, j]; g = df[df.axis == a]; ok = g.pred_axis == a
            e = (g.pred - g.act) * 1000
            ax.scatter(g.act[ok], e[ok], s=9, color=BLUE if a == 'Y' else ORG)
            bad = g[~ok]
            ax.scatter(bad.act, np.zeros(len(bad)), marker='x', s=40, color=RED, label=f'축 판정 틀림 {len(bad)}장')
            ax.axhline(0, color=SUB, lw=0.8); ax.axhspan(-90, 90, color=GRN, alpha=0.07)
            ax.set_ylim(-120 if a == 'Y' else -25, 120 if a == 'Y' else 25); ax.grid(color=GRID)
            ax.set_xlabel(f'실제 위치 {a}' + (' (mm)' if a == 'Y' else ''))
            if j == 0: ax.set_ylabel(('2차' if blk else '1차') + (' Y 오차 (µm)' if a == 'Y' else ' Θ 오차 (mΘ)'))
            ax.set_title(key, loc='left', fontsize=10.5)
            if len(bad): ax.legend(fontsize=9, frameon=False, loc='lower right')
            for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    fig.suptitle('조건별 예측 오차 (실제 위치 기준, 초록 띠 = ±0.09 의 Y 범위). 빨강 x = Y인지 Θ인지 판정이 틀린 사진', x=0.01, ha='left', fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(PROJ/'41_조건별_오차.png', dpi=100, facecolor='white'); plt.close()
    print(T.round(2).to_string())

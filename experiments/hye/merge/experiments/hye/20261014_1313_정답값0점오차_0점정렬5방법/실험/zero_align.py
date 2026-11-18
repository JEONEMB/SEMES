"""center 기준 0점 정렬: 방법 5개를 Y, Θ 각각에 독립적으로 적용.
원본(사진, 14번/15번 CSV)은 읽기만 한다. 결과는 analysis_v4/ 와 프로젝트 루트 33~35번 파일로만 쓴다.

공통 모델: x_i = k * v_i + o_i + 잡음
  x = QR 3개 평균 가로 이동 (px, center 사진 = 0), v = 명령값, o = 사진 i의 0점 어긋남 (px)
  각 방법은 o_i 를 추정하고, 보정값 x' = x - o 가 center(원점)를 지나는 직선 k*v 에 얼마나 붙는지 본다.
평가: 같은 데이터 적합(in-sample)과 짝수/홀수 교차검증(CV, 한쪽으로 맞추고 다른 쪽 채점) 둘 다."""
import os, glob, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent; PROJ = HERE.parent
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
TXT, SUB, GRID = '#1f1f1e', '#5f5e58', '#e4e3dc'
SEGCOL = ['#4a72d0', '#d2703f', '#5aa776', '#cf7aa0']

# ---------- 데이터 (읽기 전용) ----------
q = pd.read_csv(PROJ/'14_데이터_QR마커위치.csv'); f15 = pd.read_csv(PROJ/'15_데이터_파인더_볼트이동량.csv')
q['x'] = q[['dLx', 'dCx', 'dRx']].mean(1)
d = q.merge(f15[['file', 'bolt_dx']], on='file', how='left'); d.loc[d.file == 'center.bmp', 'bolt_dx'] = 0.0
def mtime(fn):
    p = ROOT/'center.bmp' if fn == 'center.bmp' else ROOT/('Theta' if fn[0] == 'T' else 'Y')/fn
    return os.path.getmtime(p)
d['t'] = d.file.map(mtime); d['t'] = (d.t - d.t.min()) / 60  # 분, center = 0
d = d.sort_values('t').reset_index(drop=True); d['order'] = np.arange(len(d))
d['ax'] = d.file.str[0].map({'Y': 'Y', 'T': 'Θ', 'c': 'C'})

def segments(g):
    """촬영 순서대로 끊긴 구간 (같은 축 안에서 3분 넘는 공백 또는 부호 바뀜) + 이동 방향."""
    g = g.sort_values('t'); seg = np.zeros(len(g), int); s = 0
    for i in range(1, len(g)):
        if g.t.iloc[i] - g.t.iloc[i-1] > 3 or np.sign(g.value.iloc[i]) != np.sign(g.value.iloc[i-1]): s += 1
        seg[i] = s
    g = g.assign(seg=seg)
    dirs = {s_: np.sign(np.polyfit(gg.t, gg.value, 1)[0]) for s_, gg in g.groupby('seg')}
    return g.assign(dir=g.seg.map(dirs))

AX = {}
for a in ['Y', 'Θ']:
    AX[a] = segments(d[d.ax == a].copy()).reset_index(drop=True)
UNIT = {'Y': ('µm', 1000 / np.polyfit(AX['Y'].value, AX['Y'].x, 1)[0]), 'Θ': ('mΘ (0.001)', 1000 / np.polyfit(AX['Θ'].value, AX['Θ'].x, 1)[0])}

def lsq(A, y): return np.linalg.lstsq(A, y, rcond=None)[0]

# ---------- 방법 5개: fit(g, train_mask) -> (pred_offset 전부, k, 설명, 파라미터 수) ----------
def m0_none(g, tr):
    k = lsq(g.value.values[tr, None], g.x.values[tr])[0]
    return np.zeros(len(g)), k, '보정 없음', 1

def m1_backlash_drift(g, tr):
    """이론: o = γ + β*방향 + a*t + c*(1-exp(-t/3분)). 예열 시간 3분은 밝기 안정 시각에서."""
    v, t, dr = g.value.values, g.t.values, g.dir.values
    A = np.c_[v, np.ones(len(g)), dr, t, 1 - np.exp(-t / 3)]
    p = lsq(A[tr], g.x.values[tr]); o = A[:, 1:] @ p[1:]
    return o, p[0], f'β={p[2]:+.2f}px, 드리프트 {p[3]:+.3f}px/분', 5

def witness_series():
    """직교축 증인: QR과 뒤쪽 볼트의 시차 비율로 '다른 축 성분'을 분리.
    q = qY + qT, b = rY*qY + rT*qT  →  qY = (b - rT q)/(rY - rT),  qT = (b - rY q)/(rT - rY)"""
    rY = lsq(AX['Y'].x.values[:, None], AX['Y'].bolt_dx.values)[0]
    rT = lsq(AX['Θ'].x.values[:, None], AX['Θ'].bolt_dx.values)[0]
    out = {}
    for a, other in [('Y', 'Θ'), ('Θ', 'Y')]:
        w = AX[other]
        if a == 'Y': e = (w.bolt_dx - rT * w.x) / (rY - rT)   # Θ 사진 속 Y 성분 = Y 0점 드리프트 (px)
        else:        e = (w.bolt_dx - rY * w.x) / (rT - rY)   # Y 사진 속 Θ 성분 = Θ 0점 드리프트 (px)
        out[a] = (w.t.values, e.values)
    return out, rY, rT
WIT, RY, RT = witness_series()

def m2_witness(g, tr, a):
    """다른 축 사진에서 잰 0점 드리프트를 시간 직선으로 맞춰 이 축 사진 시각에 적용. 이 축 자체에서는 k만 맞춘다."""
    tw, ew = WIT[a]; p = np.polyfit(np.r_[0, tw], np.r_[0, ew], 1)  # center(t=0) = 0 포함
    o = np.polyval(p, g.t.values)
    k = lsq(g.value.values[tr, None], (g.x.values - o)[tr])[0]
    return o, k, f'증인 드리프트 {p[0]:+.3f}px/분 (rY={RY:.2f}, rΘ={RT:.2f})', 1

def m3_geometry(g, tr):
    """이론: 회전 중심이 QR에서 떨어져 있으면 x = k v + c v^2 (가로 지렛대 + 깊이 변화). 0점은 center 고정."""
    v = g.value.values; A = np.c_[v, v**2]; p = lsq(A[tr], g.x.values[tr])
    return p[1] * v**2, p[0], f'2차항 c={p[1]:+.3f}px', 2

def m4_continuity(g, tr):
    """데이터: 공통 기울기 + 구간별 절편. 각 구간 직선을 v=0에서 center(0)에 맞춰 이어 붙인다."""
    v = g.value.values; S = pd.get_dummies(g.seg).values.astype(float)
    A = np.c_[v, S]; p = lsq(A[tr], g.x.values[tr]); o = S @ p[1:]
    return o, p[0], '구간 절편 ' + ', '.join(f'{b:+.2f}' for b in p[1:]) + 'px', 1 + S.shape[1]

def pelt_piecewise_linear(t, r, pen, minlen=6):
    """최적 분할(DP): 각 조각에서 r ~ a + b t 직선, 비용 = SSE + pen*조각수."""
    n = len(r); F = np.full(n + 1, np.inf); F[0] = -pen; last = np.zeros(n + 1, int)
    def sse(i, j):
        tt, rr = t[i:j], r[i:j]; A = np.c_[np.ones(j - i), tt - tt.mean()]
        res = rr - A @ lsq(A, rr); return res @ res
    for j in range(minlen, n + 1):
        for i in range(0, j - minlen + 1):
            if np.isinf(F[i]) or (i > 0 and i < minlen): continue
            c = F[i] + sse(i, j) + pen
            if c < F[j]: F[j], last[j] = c, i
    cps, j = [], n
    while j > 0: cps.append(j); j = last[j]
    return sorted(cps)[:-1]

def m5_changepoint(g, tr):
    """데이터: 촬영 순서로 잔차를 늘어놓고 변화점을 데이터가 찾게 한 뒤, 조각마다 시간 직선으로 뺀다.
    k와 변화점을 번갈아 3번 갱신. 시험 사진의 0점은 같은 조각의 학습 사진 직선에서 읽는다."""
    v, t, x = g.value.values, g.t.values, g.x.values
    k = lsq(v[tr, None], x[tr])[0]; it = np.where(tr)[0]
    for _ in range(3):
        r = x - k * v
        sig2 = np.median(np.diff(r[it])**2) / (2 * 0.455) + 1e-9  # 이웃 차분으로 잡음 분산 추정
        cps = pelt_piecewise_linear(t[it], r[it], pen=4 * sig2 * np.log(len(it)))
        bounds = [0] + cps + [len(it)]; o = np.zeros(len(g)); pid = np.zeros(len(g), int)
        pid_tr = np.searchsorted(cps, np.arange(len(it)), side='right')   # 학습 사진의 조각 번호
        pid = pid_tr[np.abs(t[:, None] - t[it][None, :]).argmin(1)]          # 시험 사진은 시간상 가장 가까운 학습 사진의 조각
        for s in range(len(bounds) - 1):
            ii = it[bounds[s]:bounds[s+1]]; ab = np.polyfit(t[ii], r[ii], 1)
            o[pid == s] = np.polyval(ab, t[pid == s])
        k = lsq(v[tr, None], (x - o)[tr])[0]
    return o, k, f'변화점 {len(cps)}개 (촬영 {", ".join(f"{t[it[b]]:.0f}" for b in cps)}분)', 1 + 2 * (len(cps) + 1)

METHODS = [('0', '보정 전', 'base', m0_none),
           ('1', '① 백래시 + 드리프트', '이론', m1_backlash_drift),
           ('2', '② 직교축 증인', '이론', None),
           ('3', '③ 회전 중심 기하 (2차)', '이론', m3_geometry),
           ('4', '④ 경계 연속 (구간 절편)', '데이터', m4_continuity),
           ('5', '⑤ 변화점 + 시간 직선', '데이터', m5_changepoint)]

def run(a, fn, tr):
    g = AX[a]
    return m2_witness(g, tr, a) if fn is None else fn(g, tr)

rows, RES = [], {}
for a in ['Y', 'Θ']:
    g = AX[a]; u_name, u = UNIT[a]; n = len(g)
    even = (np.arange(n) % 2 == 0)
    for key, name, kind, fn in METHODS:
        o, k, note, npar = run(a, fn, np.ones(n, bool))
        res = g.x.values - o - k * g.value.values
        # 좌, 우 각각 직선을 그어 v=0에서의 값 (center와 얼마나 어긋나나)
        gap = {s: np.polyfit(g.value[m], (g.x - o)[m] - k * g.value[m], 1)[1] * u for s, m in [('-', g.value < 0), ('+', g.value > 0)]}
        cv = []
        for trm in [even, ~even]:
            o2, k2, *_ = run(a, fn, trm); te = ~trm
            cv.append(g.x.values[te] - o2[te] - k2 * g.value.values[te])
        cv = np.concatenate(cv)
        RES[(a, key)] = dict(o=o, k=k, res=res, note=note)
        rows.append(dict(축=a, 방법=name, 관점=kind, 파라미터수=npar,
                         RMS_적합=np.sqrt(np.mean(res**2)) * u, 최대_적합=np.abs(res).max() * u,
                         RMS_CV=np.sqrt(np.mean(cv**2)) * u,
                         좌절편=gap['-'], 우절편=gap['+'], 좌우차=abs(gap['-'] - gap['+']), 단위=u_name, 메모=note))
tab = pd.DataFrame(rows)
tab.round(3).to_csv(PROJ/'33_결과표_0점정렬_5방법.csv', index=False, encoding='utf-8-sig')
with pd.option_context('display.width', 250, 'display.max_columns', 20):
    print(tab.drop(columns='메모').round(2)); print(tab[['축', '방법', '메모']])

# ---------- 그림: 축마다 2행 x 6열 (위: 명령값 vs 보정 후 잔차 + 촬영 경로, 아래: 촬영 시각 vs 추정 0점) ----------
for a, fname in [('Y', '34_0점정렬_Y_5방법_경로.png'), ('Θ', '35_0점정렬_Θ_5방법_경로.png')]:
    g = AX[a]; u_name, u = UNIT[a]
    fig, axs = plt.subplots(3, 6, figsize=(30, 15), gridspec_kw={'height_ratios': [1.3, 1, 1]})
    raw_res = (g.x - RES[(a, '0')]['k'] * g.value).values * u
    ylim = np.array([min(raw_res.min(), 0), max(raw_res.max(), 0)]); ylim = ylim + np.array([-1, 1]) * 0.12 * np.ptp(ylim)
    for j, (key, name, kind, fn) in enumerate(METHODS):
        R = RES[(a, key)]; res = R['res'] * u; o = R['o'] * u
        row = tab[(tab.축 == a) & (tab.방법 == name)].iloc[0]
        ax = axs[0, j]
        ax.plot(g.value, res, '-', color='#b9b7ae', lw=0.7, zorder=1)  # 촬영 순서 경로
        for s, gg in g.groupby('seg'):
            m = (g.seg == s).values; c = SEGCOL[s % 4]
            ax.scatter(g.value[m], res[m], s=16, color=c, edgecolor='white', lw=0.4, zorder=3,
                       label=f'{s+1}번째 ({gg.t.min():.0f}~{gg.t.max():.0f}분, {"+" if gg.dir.iloc[0] > 0 else "-"}방향)')
            i0, i1 = np.where(m)[0][[0, -1]]
            ax.annotate('', xy=(g.value[i0 + min(2, m.sum() - 1)], res[i0 + min(2, m.sum() - 1)]), xytext=(g.value[i0], res[i0]),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.6), zorder=4)
        ax.scatter([0], [0], marker='*', s=260, color=TXT, zorder=5, label='center')
        ax.axhline(0, color=SUB, lw=1, ls='--'); ax.axvline(0, color=SUB, lw=0.8, ls=':')
        ax.set_ylim(*ylim); ax.grid(color=GRID); ax.set_axisbelow(True)
        ax.set_title(f'{name}' + ('' if kind == 'base' else f'  [{kind}]'), loc='left', fontsize=14, color=TXT, fontweight='bold')
        ax.text(0.02, 0.97, f'RMS {row.RMS_적합:.1f} (CV {row.RMS_CV:.1f}) {u_name.split()[0]}\n좌우 0점 차 {row.좌우차:.1f}, 파라미터 {row.파라미터수}개',
                transform=ax.transAxes, va='top', fontsize=11, color=TXT, bbox=dict(fc='white', ec=GRID, alpha=0.9))
        ax.set_xlabel(f'명령한 {a}' + (' (mm)' if a == 'Y' else ''))
        if j == 0: ax.set_ylabel(f'직선(center 통과)에서 벗어난 양 ({u_name})'); ax.legend(fontsize=9, loc='lower left', frameon=False)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        if np.abs(res).max() > np.abs(ylim).max():
            ax.text(0.98, 0.03, f'일부 점이 범위 밖 (최대 {np.abs(res).max():.0f})', transform=ax.transAxes, ha='right', fontsize=10, color='#b0403a')
        z = axs[1, j]; zl = max(np.abs(res).max() * 1.15, 1e-3)
        z.plot(g.value, res, '-', color='#b9b7ae', lw=0.7, zorder=1)
        for s in g.seg.unique():
            m = (g.seg == s).values; z.scatter(g.value[m], res[m], s=14, color=SEGCOL[s % 4], edgecolor='white', lw=0.4, zorder=3)
        z.scatter([0], [0], marker='*', s=200, color=TXT, zorder=5); z.axhline(0, color=SUB, lw=1, ls='--'); z.axvline(0, color=SUB, lw=0.8, ls=':')
        z.set_ylim(-zl, zl); z.grid(color=GRID); z.set_axisbelow(True); z.set_xlabel(f'명령한 {a}' + (' (mm)' if a == 'Y' else ''))
        z.set_title(f'확대 (세로 ±{zl:.1f})', loc='left', fontsize=11, color=SUB)
        if j == 0: z.set_ylabel(f'확대: 벗어난 양 ({u_name})')
        for sp in ['top', 'right']: z.spines[sp].set_visible(False)
        b = axs[2, j]
        b.scatter(g.t, raw_res, s=8, color='#cfcdc4', zorder=1, label='보정 전 잔차')
        b.plot(g.t, o, '-', color=TXT, lw=1.6, zorder=3, label='추정 0점 o(t)')
        for s in g.seg.unique():
            m = (g.seg == s).values; b.scatter(g.t[m], o[m], s=10, color=SEGCOL[s % 4], zorder=4)
        if key == '2':
            tw, ew = WIT[a]; b.scatter(tw, ew * u, s=6, color='#8a63c4', alpha=0.6, zorder=2, label=f'증인 측정 ({"Θ" if a == "Y" else "Y"} 사진)')
        b.scatter([0], [0], marker='*', s=200, color=TXT, zorder=5)
        b.set_ylim(*ylim); b.grid(color=GRID); b.set_axisbelow(True); b.set_xlabel('촬영 시각 (분, center = 0)')
        if j == 0: b.set_ylabel(f'0점 어긋남 ({u_name})')
        if j in (0, 2): b.legend(fontsize=9, loc='lower right', frameon=False)
        b.text(0.02, 0.04, R['note'], transform=b.transAxes, fontsize=9.5, color=SUB)
        for sp in ['top', 'right']: b.spines[sp].set_visible(False)
    fig.suptitle(f'{a}: center를 기준으로 좌, 우 구간이 한 직선에 이어지도록 0점 보정 (1행: 명령값별 잔차와 촬영 경로, 같은 눈금 / 2행: 패널별 확대 / 3행: 촬영 시각별 추정 0점)',
                 x=0.01, ha='left', fontsize=17, color=TXT)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(PROJ/fname, dpi=110, facecolor='white'); plt.close()
print('saved')

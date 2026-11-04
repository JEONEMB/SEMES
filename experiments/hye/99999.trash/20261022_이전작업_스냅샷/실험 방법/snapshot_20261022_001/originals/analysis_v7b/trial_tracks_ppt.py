"""발표용: trial(촬영 구간) 6개마다 특징점 이동 경로 그림 1장 (16:9).
기준 = trial마다 첫 사진 (Y1 ±0.1, Y2 ±0.05, Θ ±0.01). center 사진은 쓰지 않는다 (문제 있음). 이동 = 각 사진 위치 - 기준 사진 위치.
위: QR 판 위에 대표 점 경로 (QR마다 4x4 격자에서 고른 점 + 판나사 2개), 사진마다 점, 끝에 화살표. 실제 크기.
왼쪽 아래: 명령값 vs 판 평균 x 이동 (한 스텝씩 일정하게 움직이는지).
오른쪽 아래: 끝값 사진에서 판 평균 이동을 뺀 나머지 (점마다 다르게 움직이는 부분) x50 화살표."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
HERE = Path(__file__).parent; PROJ = HERE.parent; V6 = PROJ/'analysis_v6'
sys.path.insert(0, str(V6))
from common import NAMES, VAL, session_of, read_raw, X0, X1, Y0, Y1
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
KY = 100 if 'amp' in sys.argv else 1                                      # 세로(y) 이동 확대 배율 ('amp' 인자)
OUT = HERE/'out'/('trial_tracks_amp' if KY > 1 else 'trial_tracks_ref'); OUT.mkdir(parents=True, exist_ok=True)
BG, FG, SUB, GRID = '#16181d', '#f2f2f2', '#a8adb7', '#3a3f4a'

F = pd.read_csv(V6/'features_qr_hp8.csv'); Z = np.load(V6/'tracks_qr_hp8.npz')
assert list(Z['names']) == NAMES
ic = NAMES.index('center.bmp'); D = Z['D_gn'].astype(np.float64); D = D - D[ic]
SES = np.array([session_of(n) for n in NAMES]); V = np.array([VAL[n] for n in NAMES])
PC = np.c_[F.x.values, F.y.values].astype(float); reg = F.region.values; isq = np.char.startswith(reg.astype(str), 'QR')
xa, xb, ya, yb = X0 - 40, X1 + 40, Y0 + 5 - 18 * (KY > 1), Y1 - 5 + 18 * (KY > 1)   # 확대 시 위아래 여유

def pick(L, P0, H=0, ok=None):
    """QR마다 6x5 격자 위치에 가까운 점을 고르되, 같은 줄에서 경로(길이 L)가 겹치지 않게. 판나사는 왼쪽/오른쪽 하나씩."""
    sc = Z['score'].mean(0) if ok is None else -ok; sel = []                    # ok: 점마다 잡음 (작을수록 우선)
    for r in ['QR좌', 'QR중', 'QR우']:
        m = np.where(reg == r)[0]
        if ok is not None: m = m[ok[m] <= np.percentile(ok[isq], 40)]          # 잡음 작은 40%만
        x0, x1 = P0[m, 0].min(), P0[m, 0].max(); y0, y1 = P0[m, 1].min(), P0[m, 1].max()
        for gy in np.linspace(y0, y1, 6):
            for gx in np.linspace(x0, x1, 5):
                j = m[np.argmin(np.hypot(P0[m, 0] - gx, P0[m, 1] - gy))]
                if all(abs(P0[j, 1] - P0[k, 1]) > 10 + H or abs(P0[j, 0] - P0[k, 0]) > L + 14 for k in sel): sel.append(j)
    for side in [P0[:, 0] < 1906, P0[:, 0] >= 1906]:
        m = np.where(~isq & side)[0]; sel.append(m[np.argmax(sc[m])])
    return np.array(sel)

INFO = {'Y1+': ('Y', '+', '0.1 mm 간격'), 'Y1-': ('Y', '-', '0.1 mm 간격'), 'Y2+': ('Y', '+', '0.05 mm 홀수칸'),
        'Y2-': ('Y', '-', '0.05 mm 홀수칸'), 'Θ+': ('Θ', '+', '0.01 간격'), 'Θ-': ('Θ', '-', '0.01 간격')}
FN = {'Y1+': 'Y1p', 'Y1-': 'Y1m', 'Y2+': 'Y2p', 'Y2-': 'Y2m', 'Θ+': 'Tp', 'Θ-': 'Tm'}

def style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=SUB, labelsize=9)
    for s in ax.spines.values(): s.set_color(GRID)

for i, s in enumerate(['Y1+', 'Y1-', 'Y2+', 'Y2-', 'Θ+', 'Θ-'], 1):
    a, sg, step = INFO[s]; unit = ' mm' if a == 'Y' else ''
    idx = np.where(SES == s)[0]; idx = idx[np.argsort(np.abs(V[idx]))]; ref = idx[0]
    vals = V[idx]; d = D[idx] - D[ref]; vmax = np.abs(vals).max()           # 기준 사진에서 0
    P0 = PC + D[ref]                                                          # 기준 사진에서의 점 위치
    crop = read_raw(NAMES[ref])[ya:yb, xa:xb]
    co = np.polyfit(vals, d.reshape(len(vals), -1), 2); fit = (np.vander(vals, 3) @ co).reshape(d.shape)   # 점마다 2차 fit
    noise = (d - fit)[..., 1].std(0)                                         # 세로 잡음
    SEL = pick(np.abs(np.median(d[-1, isq, 0])), P0, KY * np.abs(np.median(d[:, isq, 1], 1)).max(), noise if KY > 1 else None); nq = len(SEL) - 2
    cmap = plt.cm.plasma; norm = plt.Normalize(abs(vals[0]), vmax); rl = f'{a} {vals[0]:+g}'
    fig = plt.figure(figsize=(13.33, 7.5), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05 + 0.2 * (KY > 1), 1], width_ratios=[1, 1.55], left=0.05, right=0.97, top=0.86, bottom=0.08, hspace=0.32, wspace=0.12)

    # ---- 위: 대표 점 경로 ----
    ax = fig.add_subplot(gs[0, :]); style(ax)
    ax.imshow(crop, cmap='gray', extent=(xa, xb, yb, ya), vmin=-60, vmax=420)
    every = 1 if len(idx) <= 21 else 5
    for j in SEL:
        p = P0[j] + (fit if KY > 1 else d)[:, j] * np.r_[1, KY]
        if KY > 1:
            raw = P0[j] + d[:, j] * np.r_[1, KY]; ax.scatter(raw[:, 0], raw[:, 1], s=5, c=cmap(norm(np.abs(vals))), alpha=0.35, edgecolors='none', zorder=2)
        seg = np.stack([p[:-1], p[1:]], 1)
        ax.add_collection(LineCollection(seg, colors=cmap(norm(np.abs(vals[1:]))), lw=3.2, capstyle='round', zorder=3))
        k = np.arange(0, len(p), every)
        if KY == 1: ax.scatter(p[k, 0], p[k, 1], s=9, c=cmap(norm(np.abs(vals[k]))), edgecolors='none', zorder=4)
        ax.annotate('', xy=p[-1] + 7 * (p[-1] - p[-4]) / np.linalg.norm(p[-1] - p[-4]), xytext=p[-1],
                    arrowprops=dict(arrowstyle='-|>', color=cmap(1.0), lw=0, mutation_scale=13), zorder=5)
        ax.scatter(*P0[j], s=34, c='white', edgecolors='black', lw=0.8, zorder=6)
    ax.set_xlim(xa, xb); ax.set_ylim(yb, ya); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_anchor('W')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.008); cb.ax.tick_params(colors=SUB, labelsize=9); cb.outline.set_edgecolor(GRID)
    cb.set_label(f'|{a}|{unit}', color=SUB, fontsize=10)
    ax.set_title(f'흰 점 = 기준 사진 ({rl}) 위치, 선 = 나머지 사진 {len(idx) - 1}장 동안의 경로 ' + ('(실제 크기)' if KY == 1 else f'(가로 실제 크기, 세로 x{KY})') + f', 대표 점 {len(SEL)}개 (QR {nq} + 판나사 2, 겹치지 않게' + (', 세로 잡음 작은 점' if KY > 1 else '') + ' 고름)' + ('\n선 = 점마다 2차 fit, 옅은 점 = 실제 측정' if KY > 1 else ''),
                 loc='left', color=SUB, fontsize=10.5)

    # ---- 왼쪽 아래: 명령값 vs 판 평균 이동 ----
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    mx = np.median(d[:, isq, 0], 1); my = np.median(d[:, isq, 1], 1)
    ax.plot(vals, mx, color='#5ab0ff', lw=2, zorder=2); ax.scatter(vals, mx, c=cmap(norm(np.abs(vals))), s=26, zorder=3, edgecolors='none')
    ax.set_xlabel(f'명령값 {a}{unit}', color=SUB, fontsize=10); ax.set_ylabel(f'판 평균 x 이동 (px, 기준 {rl})', color=SUB, fontsize=10)
    ax.grid(color=GRID, lw=0.6); ax.axhline(0, color=GRID)
    slope = np.polyfit(vals, mx, 1)[0]
    ax.text(0.03 if sg == '+' else 0.97, 0.92 if sg == '+' else 0.08, f'x 기울기 {slope:+.1f} px / {"mm" if a == "Y" else "Θ 1"}\n끝값 y 이동 {my[-1]:+.2f} px',
            transform=ax.transAxes, ha='left' if sg == '+' else 'right', va='top' if sg == '+' else 'bottom', color=FG, fontsize=11)
    ax.set_title('한 스텝씩 일정하게 이동 (거의 x 방향만)', loc='left', color=FG, fontsize=12)

    # ---- 오른쪽 아래: 판 평균 이동 뺀 나머지 ----
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    ax.imshow(crop, cmap='gray', extent=(xa, xb, yb, ya), vmin=-150, vmax=600)
    r = d[-1] - np.median(d[-1, isq], 0); AMP = 50; mag = np.hypot(*r.T)
    cm2 = plt.cm.cool; n2 = plt.Normalize(0, np.percentile(mag[isq], 98))
    ax.quiver(P0[isq, 0], P0[isq, 1], r[isq, 0], r[isq, 1], color=cm2(n2(mag[isq])), angles='xy', scale_units='xy', scale=1 / AMP,
              width=0.0028, headwidth=3.5, headlength=4, zorder=3)
    ax.set_xlim(xa, xb); ax.set_ylim(yb, ya); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'끝값 사진 ({vals[-1]:+g}): 기준 대비, 평균 이동 뺀 나머지 x{AMP}  (QR 점 232개, 중앙값 {np.median(mag[isq]):.2f} px)',
                 loc='left', color=FG, fontsize=12)

    fig.text(0.05, 0.945, f'trial {i}   {s}', color=FG, fontsize=24, fontweight='bold', va='center')
    fig.text(0.215, 0.945, f'{a} {sg}방향, {step}, 기준 {rl}{unit}, 나머지 {len(idx) - 1}장 (~ {vals[-1]:+g}{unit})', color=SUB, fontsize=15, va='center')
    fig.savefig(OUT/f'{i}_trial_{FN[s]}_특징점경로.png', dpi=200, facecolor=BG); plt.close(fig)
    print(s, f'{slope:+.2f}')

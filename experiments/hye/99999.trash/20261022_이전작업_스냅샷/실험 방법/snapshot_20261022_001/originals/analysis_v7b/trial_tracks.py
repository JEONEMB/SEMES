"""trial(촬영 구간) 6개마다 특징점 이동 경로를 선으로 이어 그린다. 사진 6장.
점: v6 hp8 추적 (QR 232개 + 판나사 18개), center 기준 이동 D.
위: center 사진 위에 실제 크기 경로. 아래: 사진마다 판 전체 평균 이동을 빼고 x30 확대 (점마다 다르게 움직이는 부분)."""
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
OUT = HERE/'out'/'trial_tracks'; OUT.mkdir(parents=True, exist_ok=True)

F = pd.read_csv(V6/'features_qr_hp8.csv'); Z = np.load(V6/'tracks_qr_hp8.npz')
assert list(Z['names']) == NAMES
ic = NAMES.index('center.bmp'); D = Z['D_gn'].astype(np.float64); D = D - D[ic]
SES = np.array([session_of(n) for n in NAMES]); V = np.array([VAL[n] for n in NAMES])
cen = read_raw('center.bmp')
PAD = 100; xa, xb, ya, yb = X0 - PAD, X1 + PAD, Y0 - PAD, Y1 + PAD
crop = cen[ya:yb, xa:xb]
isq = F.region.str.startswith('QR').values
AMP = 30
TITLE = {'Y1+': 'Y1+ (Y 0.1 간격, +방향)', 'Y1-': 'Y1- (Y 0.1 간격, -방향)', 'Y2+': 'Y2+ (Y 0.05 홀수칸, +방향)',
         'Y2-': 'Y2- (Y 0.05 홀수칸, -방향)', 'Θ+': 'Θ+ (Θ 0.01 간격, +방향)', 'Θ-': 'Θ- (Θ 0.01 간격, -방향)'}
FN = {'Y1+': 'Y1p', 'Y1-': 'Y1m', 'Y2+': 'Y2p', 'Y2-': 'Y2m', 'Θ+': 'Tp', 'Θ-': 'Tm'}

def tracks(P, ax, cmap, vals, lw, alpha):
    """P: (사진, 점, 2) 경로. 점마다 center에서 출발해 명령값 순서로 잇고, 구간 색 = 명령값."""
    norm = plt.Normalize(0, np.abs(vals).max())
    for j in range(P.shape[1]):
        seg = np.stack([P[:-1, j], P[1:, j]], 1)
        lc = LineCollection(seg, cmap=cmap, norm=norm, lw=lw, alpha=alpha); lc.set_array(np.abs(vals[1:])); ax.add_collection(lc)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)

for i, s in enumerate(['Y1+', 'Y1-', 'Y2+', 'Y2-', 'Θ+', 'Θ-'], 1):
    idx = np.where(SES == s)[0]; idx = idx[np.argsort(np.abs(V[idx]))]
    idx = np.r_[ic, idx]; vals = V[idx]                                   # center(0) -> 작은 값 -> 큰 값
    d = D[idx]; p0 = np.c_[F.x.values, F.y.values][None].astype(float)
    unit = 'mm' if s[0] == 'Y' else '(Θ 단위)'
    fig, axs = plt.subplots(2, 1, figsize=(12, 9.6), gridspec_kw=dict(height_ratios=[1, 1]))

    ax = axs[0]
    ax.imshow(crop, cmap='gray', extent=(xa, xb, yb, ya), vmin=0, vmax=255)
    sm = tracks(p0 + d, ax, 'plasma', vals, 0.9, 0.85)
    ax.scatter(F.x[isq], F.y[isq], s=5, c='cyan', lw=0, label='QR 점 (center 위치)')
    ax.scatter(F.x[~isq], F.y[~isq], s=14, c='lime', marker='s', lw=0, label='판나사 점 (center 위치)')
    ax.set_xlim(xa, xb); ax.set_ylim(yb, ya); ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
    ax.set_title(f'(a) 실제 크기 경로: 점 {len(F)}개, center에서 출발해 명령값 순서로 연결. 끝값에서 평균 이동 '
                 f'dx {np.median(d[-1, :, 0]):+.1f}px, dy {np.median(d[-1, :, 1]):+.2f}px', loc='left', fontsize=11)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01); cb.set_label(f'|명령값| {unit}')

    ax = axs[1]
    r = d - np.median(d, axis=1, keepdims=True)                           # 사진마다 판 평균 이동 제거
    ax.imshow(crop, cmap='gray', extent=(xa, xb, yb, ya), vmin=0, vmax=255, alpha=0.45)
    sm = tracks(p0 + AMP * r, ax, 'viridis', vals, 0.9, 0.9)
    ax.scatter(F.x[isq], F.y[isq], s=5, c='red', lw=0)
    ax.scatter(F.x[~isq], F.y[~isq], s=14, c='orange', marker='s', lw=0)
    ax.set_xlim(xa, xb); ax.set_ylim(yb, ya); ax.set_aspect('equal')
    rm = np.hypot(r[-1, :, 0], r[-1, :, 1])
    ax.set_title(f'(b) 판 평균 이동을 뺀 나머지 x{AMP} 확대: 점마다 다르게 움직이는 부분. 끝값에서 크기 중앙값 {np.median(rm):.2f}px, 최대 {rm.max():.2f}px',
                 loc='left', fontsize=11)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01); cb.set_label(f'|명령값| {unit}')

    fig.suptitle(f'trial {i}: {TITLE[s]}, 사진 {len(idx) - 1}장, 명령값 {vals[1]:+g} ~ {vals[-1]:+g} {unit}',
                 x=0.01, ha='left', fontsize=14)
    fig.tight_layout(); fig.savefig(OUT/f'{i}_trial_{FN[s]}_특징점경로.png', dpi=130); plt.close(fig)
    print(s, len(idx) - 1)

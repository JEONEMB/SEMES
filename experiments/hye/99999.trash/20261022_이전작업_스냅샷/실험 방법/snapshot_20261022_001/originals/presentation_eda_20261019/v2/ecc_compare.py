# ECC 기준 패치 비교: 예전 고정 창(ecc_motion_oldbox.csv) vs 라벨 기준(ecc_motion.csv), 같은 원본 281장
#   출력: img/ecc_compare.csv (항목별 표), img/ecc_compare.png (사진별 이동량 차이)
import pathlib, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = pathlib.Path(__file__).parent; IMG = HERE / 'img'
for f in ['NotoSansKR-Regular.ttf', 'NotoSansKR-Medium.ttf']: fm.fontManager.addfont(f'C:/Windows/Fonts/{f}')
plt.rcParams.update({'font.family': 'Noto Sans KR', 'font.size': 8, 'axes.unicode_minus': False})
INK, SUB, GRID, BG = '#111827', '#4B5563', '#E5E7EB', '#F7F7F8'

def trial(f):
    if f == 'center.bmp': return 'center'
    v = float(f[1:-4]); s = '+' if v > 0 else '-'
    return ('Θ ' + s) if f[0] == 'T' else (('Y 0.1 간격 ' if round(abs(v) * 100) % 10 == 0 else 'Y 0.05 홀수 ') + s)

A = pd.read_csv(IMG / 'ecc_motion_oldbox.csv').set_index('file'); B = pd.read_csv(IMG / 'ecc_motion.csv').set_index('file')
C = pd.read_csv(IMG / 'photo_motion.csv').set_index('file')
A, B = A.drop('center.bmp'), B.drop('center.bmp'); C = C.loc[A.index]
def stats(M):
    M = M.copy(); M['trial'] = M.index.map(trial); out = {}
    for t, g in M.groupby('trial'):
        k, b = np.polyfit(g.value, g.dx, 1); out[t] = (k, np.std(g.dx - (k * g.value + b), ddof=2))
    cc = np.concatenate([M[f'{q}_cc'] for q in 'LCR'])
    return dict(cc_min=cc.min(), cc_med=np.median(cc), ncc_min=min(M[f'{q}_ncc'].min() for q in 'LCR'),
                slope_Y=(min(v[0] for t, v in out.items() if t[0] == 'Y'), max(v[0] for t, v in out.items() if t[0] == 'Y')),
                slope_Tp=out['Θ +'][0], slope_Tm=out['Θ -'][0],
                resid_Y=(min(v[1] for t, v in out.items() if t[0] == 'Y'), max(v[1] for t, v in out.items() if t[0] == 'Y')),
                resid_T=(min(v[1] for t, v in out.items() if t[0] == 'Θ'), max(v[1] for t, v in out.items() if t[0] == 'Θ')),
                vsC_dx=(M.dx - C.dx).abs().max(), vsC_dy=(M.dy - C.dy).abs().max(),
                vsC_dx_sd=(M.dx - C.dx).std(), vsC_dy_sd=(M.dy - C.dy).std())
sa, sb = stats(A), stats(B)
d = pd.DataFrame({'dx': B.dx - A.dx, 'dy': B.dy - A.dy}); dq = {q: (B[f'd{q}x'] - A[f'd{q}x']) for q in 'LCR'}
f2 = lambda r: f'{r[0]:.3f} ~ {r[1]:.3f}'
rows = [('ECC 상관계수 최저', f"{sa['cc_min']:.4f}", f"{sb['cc_min']:.4f}"), ('ECC 상관계수 중앙값', f"{sa['cc_med']:.4f}", f"{sb['cc_med']:.4f}"),
        ('Y 기울기 (px/mm)', f2(sa['slope_Y']), f2(sb['slope_Y'])), ('Θ 기울기 +쪽 / -쪽', f"{sa['slope_Tp']:.3f} / {sa['slope_Tm']:.3f}", f"{sb['slope_Tp']:.3f} / {sb['slope_Tm']:.3f}"),
        ('직선 잔차 Y (px)', f2(sa['resid_Y']), f2(sb['resid_Y'])), ('직선 잔차 Θ (px)', f2(sa['resid_T']), f2(sb['resid_T'])),
        ('특징점과 차이 가로 최대 (px)', f"{sa['vsC_dx']:.4f}", f"{sb['vsC_dx']:.4f}"), ('특징점과 차이 가로 σ (px)', f"{sa['vsC_dx_sd']:.4f}", f"{sb['vsC_dx_sd']:.4f}"),
        ('특징점과 차이 세로 최대 (px)', f"{sa['vsC_dy']:.4f}", f"{sb['vsC_dy']:.4f}")]
T = pd.DataFrame(rows, columns=['항목', '고정 창', '라벨 기준']); T.to_csv(IMG / 'ecc_compare.csv', index=False, encoding='utf-8-sig')
print(T.to_string(index=False))
print('사진별 두 방법 차이 (QR 3개 평균): 가로 max %.4f sd %.4f mean %+.4f | 세로 max %.4f sd %.4f mean %+.4f' % (d.dx.abs().max(), d.dx.std(), d.dx.mean(), d.dy.abs().max(), d.dy.std(), d.dy.mean()))
for q in 'LCR': print(q, '가로 차이 max %.4f mean %+.4f' % (dq[q].abs().max(), dq[q].mean()))

# 그림: 사진별 두 방법의 가로 이동 차이 (명령값 순), 특징점과의 차이도 같이
M = A.copy(); M['trial'] = M.index.map(trial); M['ddx'] = d.dx; M['a_c'] = A.dx - C.dx; M['b_c'] = B.dx - C.dx
fig, ax = plt.subplots(figsize=(6.0, 2.6), dpi=250); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(colors=SUB, length=2, labelsize=7)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
x = np.arange(len(M)); order = M.sort_values(['trial', 'value']).index
ax.plot(x, M.loc[order, 'a_c'], 'o', ms=1.6, color='#9CA3AF', label='고정 창 − 특징점')
ax.plot(x, M.loc[order, 'b_c'], 'o', ms=1.6, color='#0071E3', label='라벨 기준 − 특징점')
ax.plot(x, M.loc[order, 'ddx'], '-', lw=0.8, color='#D9480F', label='라벨 기준 − 고정 창')
ax.axhline(0, color=INK, lw=0.6); ax.set_ylabel('가로 이동 차이 (px)', color=SUB, fontsize=7)
ax.set_xlabel('사진 (trial, 명령값 순)', color=SUB, fontsize=7); ax.set_xticks([])
ax.legend(fontsize=6, frameon=False, loc='upper left', ncol=3)
fig.tight_layout(pad=0.4); fig.savefig(IMG / 'ecc_compare.png', facecolor=BG); plt.close(fig)

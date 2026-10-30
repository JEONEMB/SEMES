"""Y 사진(위 줄)과 Θ 사진(아래 줄)을 나란히 놓고, QR 위치가 같아 구분 안 되는 짝을 선으로 잇는다."""
import numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
P = Path(__file__).parent.parent
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
ORG, GRN, LINE, TXT, SUB = '#ED7D31', '#92D050', '#1F4E78', '#1f1f1e', '#5f5e58'
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
im = cv2.imdecode(np.fromfile(str(ROOT/'center.bmp'), np.uint8), cv2.IMREAD_GRAYSCALE)
thumb = cv2.resize(im[200:2000, 1100:2700], (120, 135), interpolation=cv2.INTER_AREA)
# 판정 결과 모으기 -----------------------------------------------------------
S = 'C:/Users/SSAFY/AppData/Local/Temp/claude/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES/0478c349-0ea5-4fa5-b878-4da27eb190c2/scratchpad/'
Yo, To = pd.read_pickle(S+'Yo.pkl'), pd.read_pickle(S+'To.pkl')          # v1
J = pd.read_pickle(P/'analysis_v2/judge.pkl'); r2 = J['res']['v2-a2 (판 전체 원근 변환 14개)']
Ys, Ts = J['Ys'], J['Ts']
def twins(ys, ts, yflag, tflag):
    """yflag/tflag: 사진별 구분 불가 여부. 반환: (Y값, 짝Θ값) 목록, (Θ값, 짝Y값) 목록"""
    tm, tv = ts.m.values, ts.value.values; ym, yv = ys.m.values, ys.value.values
    YT = [(v, float(np.interp(m, tm, tv))) for v, m, f in zip(yv, ym, yflag) if f]
    TY = [(v, float(np.interp(m, ym, yv))) for v, m, f in zip(tv, tm, tflag) if f]
    return YT, TY
v1 = twins(Ys, Ts, (Ys.file.map(dict(zip(Yo.파일명, Yo.판정))) == '구분 불가').values,
                   (Ts.file.map(dict(zip(To.파일명, To.판정))) == '구분 불가').values)
v2 = twins(Ys, Ts, (r2['out']['Y'].규칙2 == '불가').values, (r2['out']['Θ'].규칙2 == '불가').values)
# 그리기 ---------------------------------------------------------------------
W = 100.0; xY = lambda v: W / 2 + v / 2.0 * (W / 2); xT = lambda v: W / 2 + v / 1.0 * (W / 2)
yv = np.round(np.r_[np.arange(-2, 0, 0.05), 0, np.arange(0.05, 2.001, 0.05)], 2)
tv = np.round(np.r_[np.arange(-1, 0, 0.01), 0, np.arange(0.01, 1.0001, 0.01)], 2)
fig, axes = plt.subplots(2, 1, figsize=(22, 13))
for ax, (YT, TY), title in [(axes[0], v1, f'v1: QR 네모 가로 위치만 → 구분 안 되는 짝  Y 쪽 {len(v1[0])}세트, Θ 쪽 {len(v1[1])}세트'),
                            (axes[1], v2, f'v2-a2: QR 판 전체 원근 변환 (QR만 쓰는 방법 중 최고) → Y 쪽 {len(v2[0])}세트, Θ 쪽 {len(v2[1])}세트')]:
    ax.set_xlim(-2, W + 2); ax.set_ylim(-3.2, 12.6); ax.axis('off')
    ax.set_title(title, loc='left', fontsize=15, color=TXT, pad=6)
    yT, yB, h = 8.0, 0.5, 2.4                                             # 위 줄 바닥, 아래 줄 바닥, 상자 높이
    badY = {round(a, 2) for a, _ in YT}; badT = {round(a, 2) for a, _ in TY}
    wy = W / len(yv) * 0.82; wt = W / len(tv) * 0.78
    for v in yv:
        if v == 0: continue
        bad = round(v, 2) in badY
        ax.add_patch(Rectangle((xY(v) - wy / 2, yT), wy, h, fc=ORG, ec='#9C4A12' if bad else 'none', lw=1.4, alpha=1 if bad else 0.35))
    for v in tv:
        if v == 0: continue
        bad = round(v, 2) in badT
        ax.add_patch(Rectangle((xT(v) - wt / 2, yB), wt, h, fc=GRN, ec='#4E7A1E' if bad else 'none', lw=1.0, alpha=1 if bad else 0.35))
    for yy in (yT, yB):
        ax.add_artist(AnnotationBbox(OffsetImage(thumb, zoom=0.17, cmap='gray'), (W / 2, yy + h / 2), frameon=True, pad=0.05,
                                     bboxprops=dict(ec=LINE, lw=1.2)))
    for a, b in YT: ax.plot([xY(a), xT(b)], [yT, yB + h], color=LINE, lw=1.0, alpha=0.75, zorder=1)
    for a, b in TY: ax.plot([xT(a), xY(b)], [yB + h, yT], color='#C00000', lw=0.8, alpha=0.5, ls=(0, (3, 2)), zorder=1)
    for v in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]: ax.text(xY(v), yT + h + 0.35, f'Y {v:+g}', ha='center', fontsize=10, color=SUB)
    for v in [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]: ax.text(xT(v), yB - 0.8, f'Θ {v:+g}', ha='center', fontsize=10, color=SUB)
    ax.text(-2, yT + h + 1.3, 'Y 사진 80장 (Θ = 0)  + center', fontsize=12, color='#9C4A12')
    ax.text(-2, yB - 2.2, 'Θ 사진 200장 (Y = 0)  + center', fontsize=12, color='#4E7A1E')
    lo = min(a for a, _ in YT); hi = max(a for a, _ in YT)
    ax.annotate('', xy=(xY(lo), yT + h + 0.95), xytext=(xY(hi), yT + h + 0.95), arrowprops=dict(arrowstyle='<->', color=TXT, lw=1))
    ax.text(W / 2, yT + h + 1.25, f'구분 안 되는 Y: {lo:+.2f} ~ {hi:+.2f}mm', ha='center', fontsize=11, color=TXT)
fig.text(0.01, 0.012, '진한 상자 = 구분 안 되는 사진, 흐린 상자 = 구분되는 사진. 파란 실선 = Y 사진 → QR 위치가 같은 Θ 지점, 빨간 점선 = Θ 사진 → QR 위치가 같은 Y 지점. '
         'Y 1mm = QR 14.3px, Θ 1 = 80px라서 Y ±2mm는 Θ ±0.36과 겹치고, 그 밖의 Θ 사진은 짝이 없다.', fontsize=10.5, color=SUB)
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(P/'27_구분안되는짝_연결도.png', dpi=120, facecolor='white')
print('v1', len(v1[0]), len(v1[1]), '| v2-a2', len(v2[0]), len(v2[1]))

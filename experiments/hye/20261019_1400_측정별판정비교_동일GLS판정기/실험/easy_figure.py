# 쉽게 보는 그림: 280장 판정에서 한 번에 몇 장 틀리나 (반복 5분할 20회 평균)
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ['NotoSansKR-Regular.ttf', 'NotoSansKR-Bold.ttf']: fm.fontManager.addfont(f'C:/Windows/Fonts/{f}')
plt.rcParams.update({'font.family': 'Noto Sans KR', 'axes.unicode_minus': False})
S = pd.read_csv('summary.csv').set_index('method')
items = [('ref205_P12', '기존 방법 (특징점 205개)'), ('ecc_affine18', 'ECC (QR 3개 통째로)'),
         ('corner_nondiag_P12', '모서리 630개 (대각선 빼고)'), ('corner_valid_P12', '모서리 중 깨끗한 점만'), ('corner_all_P12', '모서리 755개 전부')]
labels = [l for _, l in items]; vals = [S.loc[k, 'rep_wrong_per_round'] for k, _ in items]
best = min(vals)
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200); fig.patch.set_facecolor('white')
y = range(len(items))[::-1]
cols = ['#0071E3' if v == best else ('#D9480F' if v > 9 else '#9CA3AF') for v in vals]
ax.barh(list(y), vals, color=cols, height=0.55)
for yy, v in zip(y, vals): ax.text(v + 0.2, yy, f'{v:.1f}장', va='center', fontsize=12, color='#111827', fontweight='bold')
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=12, color='#111827')
ax.set_xlim(0, 13); ax.set_xticks([]); [ax.spines[s].set_visible(False) for s in ax.spines]
ax.tick_params(length=0)
fig.suptitle('사진 280장을 Y인지 Θ인지 맞히게 했을 때, 틀린 장수', x=0.02, ha='left', fontsize=14, fontweight='bold', color='#111827')
ax.set_title('(새 사진으로 시험하는 걸 20번 반복한 평균)', loc='left', fontsize=10, color='#4B5563', pad=6)
ax.axvspan(5.5, 6.4, color='#F3F4F6', zorder=0)
ax.text(5.95, -0.75, '거의 다 6장 정도', ha='center', fontsize=10, color='#4B5563')
fig.tight_layout(); fig.savefig('쉽게보기_틀린장수.png', facecolor='white')
# 슬라이드용: 제목은 슬라이드 제목으로 쓰고, 바탕은 슬라이드 회색과 같게
fig._suptitle.set_visible(False); ax.set_title('', loc='left'); fig.patch.set_facecolor('#F7F7F8'); ax.set_facecolor('#F7F7F8')
fig.tight_layout(); fig.savefig('쉽게보기_틀린장수_슬라이드.png', facecolor='#F7F7F8'); print('ok')

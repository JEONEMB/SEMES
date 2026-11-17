# trial 6개마다: (위) QR 판 위 755점 지도  초록 = 그 trial 모든 사진에서 통과, 주황 = 한 장이라도 불확실
#               (아래) 사진마다 통과/불확실 점 수 막대 (trial의 모든 사진 = 전수)
import numpy as np, pandas as pd, pathlib, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image, ImageDraw
HERE = pathlib.Path(__file__).parent; OUT = HERE / 'out'
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
for f in ['NotoSansKR-Regular.ttf', 'NotoSansKR-Medium.ttf']: fm.fontManager.addfont(f'C:/Windows/Fonts/{f}')
plt.rcParams.update({'font.family': 'Noto Sans KR', 'font.size': 8, 'axes.unicode_minus': False})
GREEN, ORANGE, SUB, GRID, BG = (22, 163, 74), (234, 88, 12), '#4B5563', '#E5E7EB', '#F7F7F8'
D = pd.read_csv(OUT / 'corners_280.csv')
def trial(f):
    if f == 'center.bmp': return 'center'
    v = float(f[1:-4]); s = '+' if v > 0 else '-'
    if f[0] == 'T': return 'Θ ' + s
    return ('Y 0.1 간격 ' if round(abs(v) * 100) % 10 == 0 else 'Y 0.05 홀수 ') + s
D['trial'] = D.file.map(trial); D['value'] = D.file.map(lambda f: 0.0 if f == 'center.bmp' else float(f[1:-4]))
C = D[D.trial == 'center'].set_index('id')
X0, Y0, X1, Y1, K = 1545, 962, 2262, 1180, 3
base = cv2.imdecode(np.fromfile(str(ROOT / 'center.bmp'), np.uint8), cv2.IMREAD_GRAYSCALE)[Y0:Y1, X0:X1]
base = Image.fromarray(cv2.resize(base, None, fx=K, fy=K, interpolation=cv2.INTER_CUBIC)).convert('RGB')
TRIALS = ['Y 0.1 간격 +', 'Y 0.1 간격 -', 'Y 0.05 홀수 +', 'Y 0.05 홀수 -', 'Θ +', 'Θ -']
rows = []
for i, t in enumerate(TRIALS, 1):
    T = D[D.trial == t]
    ok_all = T.groupby('id').valid.all(); n_bad = (~T.valid).groupby(T.id).sum(); n_ph = T.file.nunique()
    im = base.copy(); dr = ImageDraw.Draw(im); r = 7   # 모듈 한 칸(약 7.4px x 3배) 보다 작게 해서 이웃 네모가 안 겹치게
    for cid, ok in ok_all.items():
        x, y = (C.loc[cid, 'x'] - X0) * K, (C.loc[cid, 'y'] - Y0) * K
        col = GREEN if ok else ORANGE
        dr.rectangle((x - r - 2, y - r - 2, x + r + 2, y + r + 2), outline=(15, 23, 42), width=2)
        dr.rectangle((x - r, y - r, x + r, y + r), outline=col, width=3)
    im.save(OUT / f'map_{i}.png')
    # 사진마다 통과/불확실 수
    P = T.groupby(['file', 'value']).valid.agg(['sum', 'size']).reset_index().sort_values('value', key=abs)
    fig, ax = plt.subplots(figsize=(11.6, 1.35), dpi=250); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    xs = np.arange(len(P)); w = 0.8
    ax.bar(xs, P['sum'], w, color='#%02x%02x%02x' % GREEN, label='통과')
    ax.bar(xs, P['size'] - P['sum'], w, bottom=P['sum'], color='#%02x%02x%02x' % ORANGE, label='불확실')
    step = 1 if len(P) <= 20 else 5
    ax.set_xticks(xs[step - 1::step] if step > 1 else xs); ax.set_xticklabels([f'{v:+.2f}' for v in P.value.values[step - 1::step]] if step > 1 else [f'{v:+.2f}' for v in P.value], fontsize=6.5, color=SUB)
    ax.set_ylim(0, 760); ax.set_yticks([0, 250, 500, 755]); ax.tick_params(colors=SUB, length=2, labelsize=6.5)
    ax.set_ylabel('점 수', color=SUB, fontsize=7); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
    ax.set_xlim(-0.6, len(P) - 0.4); ax.legend(ncol=2, frameon=False, fontsize=7, loc='lower right', bbox_to_anchor=(1, 1.0))
    fig.tight_layout(pad=0.3); fig.savefig(OUT / f'bar_{i}.png', facecolor=BG); plt.close(fig)
    rows.append(dict(trial=t, photos=n_ph, always_ok=int(ok_all.sum()), sometimes_bad=int(((n_bad > 0) & (n_bad < n_ph)).sum()),
                     always_bad=int((n_bad == n_ph).sum()), valid_rate=round(T.valid.mean() * 100, 2),
                     per_photo_min=int(P['sum'].min()), per_photo_max=int(P['sum'].max())))
S = pd.DataFrame(rows); S.to_csv(OUT / 'trial_summary.csv', index=False, encoding='utf-8-sig'); print(S.to_string(index=False))
c = D[D.trial == 'center']; print('center 통과', int(c.valid.sum()), '/', len(c))
r = D[~D.valid].reason.str.split(';').explode().value_counts(); print('불확실 이유 (전체 사진, 점-사진 단위):'); print(r)
print('검정 이웃 수별 통과율:'); print(D.groupby('black_neighbors').valid.mean().round(3))

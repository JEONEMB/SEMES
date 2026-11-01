# 보고용 ROI 그림 (한국어 글자) : C++ 결과 roi_candidates.csv / roi_selected.yml 을 읽어 그리기만 함
import cv2, csv, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
D, IMG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(fp); plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
C = list(csv.DictReader(open(f'{D}/roi_candidates.csv', encoding='utf-8-sig')))
fs = cv2.FileStorage(f'{D}/roi_selected.yml', cv2.FILE_STORAGE_READ); R = fs.getNode('rois')
sel = [R.at(i).getNode('name').string() for i in range(R.size())]
col = {'door_panel': '#2e9d5f', 'bottom': '#8a8a8a', 'bracket_upper': '#e08a00', 'bracket_lower': '#c25e00', 'ledge': '#a64fc4', 'wall': '#3a6fd8'}
nm = {'door_panel': '하단 문', 'bottom': '문 아래 바닥(경첩)', 'bracket_upper': '돌출부 · 브래킷 위', 'bracket_lower': '돌출부 · 플레이트 아래', 'ledge': '돌출부 · 벽 선반', 'wall': '나머지 벽'}
ref = cv2.imread(f'{IMG}/center.png', 0)
fig, ax = plt.subplots(figsize=(16, 9.6), dpi=110)
ax.imshow(ref, cmap='gray', vmin=0, vmax=255); ax.set_xlim(0, 3840); ax.set_ylim(2160, 0)
for q in (1650, 1907, 2165):
    ax.add_patch(plt.Rectangle((q - 110, 955), 220, 220, fill=False, ec='white', lw=2.4))
ax.text(1907, 1205, 'QR 3개 (s)', color='white', ha='center', va='top', fontsize=10, fontweight='bold', bbox=dict(fc='#1f6f78', ec='none', pad=1.5))
for c in C:
    x, y, h = int(float(c['cx'])), int(float(c['cy'])), int(float(c['hw']))
    s = c['name'] in sel
    ax.add_patch(plt.Rectangle((x - h, y - h), 2 * h, 2 * h, fill=False, ec=col[c['zone']], lw=3.2 if s else 1.1, alpha=1 if s else .8))
for c in C:
    if c['name'] in sel:
        x, y, h = int(float(c['cx'])), int(float(c['cy'])), int(float(c['hw']))
        ax.add_patch(plt.Rectangle((x - h - 14, y - h - 14), 2 * h + 28, 2 * h + 28, fill=False, ec='#ffd400', lw=3.4))
        ax.text(x, y - h - 26, f"선택 : {c['name'].replace('wall_', '')}  ({x}, {y})  120×120", color='black', ha='center', va='bottom', fontsize=10, fontweight='bold', bbox=dict(fc='#ffd400', ec='none', pad=2))
hs = [Line2D([], [], color=col[z], lw=2.4, label=f"{nm[z]} ({sum(c['zone'] == z for c in C)})") for z in col]
hs += [Line2D([], [], color='#ffd400', lw=3.4, label='선택된 ROI (d = s − 평균)'), Line2D([], [], color='white', lw=2.4, label='QR 3개')]
leg = ax.legend(handles=hs, loc='upper left', fontsize=9.5, framealpha=.92)
ax.set_title(f'ROI 후보 {len(C)}곳과 최적 ROI — 1차 첫 번째 측정 240장으로 선택', fontsize=14)
fig.tight_layout(); fig.savefig(OUT); print('saved', OUT, sel)

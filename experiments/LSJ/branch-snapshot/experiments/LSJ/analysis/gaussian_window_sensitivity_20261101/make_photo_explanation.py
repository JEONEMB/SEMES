"""Scientific figures from the original photo and saved fits; no refitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                     'font.size': 11, 'savefig.dpi': 150})
manifest = json.loads((HERE.parent/'reference_line_methods_108_20261030/configs/inputs_manifest.json').read_text(encoding='utf-8'))
ref = manifest['references'][0]
source = Path(ref['path'])
assert hashlib.sha256(source.read_bytes()).hexdigest() == ref['sha256']
photo = np.array(Image.open(source).convert('RGB'))
examples = json.loads((HERE/'example_provenance.json').read_text(encoding='utf-8'))
example = next(e for e in examples if e['target'] == 'lower_structure')
x, cy = example['x'], example['coarse_y']
z = np.load(HERE/'paired_arrays.npz')
key = 'R01_lower_structure'
ix = int(np.flatnonzero(z[key+'_xs'] == x)[0])
profile = next(p for p in json.loads((HERE/'profiles.json').read_text())
               if (p['ref'], p['target'], p['x']) == ('R01', 'lower_structure', x))
values = np.array(profile['sobel_samples_m10_p10'])
ns = [5, 11, 13]
blue, yellow, red = '#107bc1', '#ffd000', '#d82c45'
results = []
for n in ns:
    ni = (n-3)//2
    mu = float(z[key+'_mu'][ix, ni, 0])
    assert abs(mu-example['mu_by_n'][str(n)]) < 1e-12
    results.append({'n': n, 'mu_y_px': mu,
                    'A': float(z[key+'_A'][ix, ni, 0]),
                    'sigma_px': float(z[key+'_sigma'][ix, ni, 0])})

# The lower row repeats exactly the same crop. Marks are coordinate annotations.
fig = plt.figure(figsize=(11, 10), facecolor='white')
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], left=.075, right=.97,
                      top=.91, bottom=.17, hspace=.30, wspace=.27)
overview = fig.add_subplot(gs[0, :])
overview.imshow(photo)
overview.set_axis_off()
overview.add_patch(Rectangle((x-75, cy-40), 150, 80, fill=False, ec=yellow, lw=2))
overview.plot(x, cy, '+', color=yellow, ms=9, mew=2)
overview.annotate('이 하단 경계의 한 위치를 확대', (x-75, cy), xytext=(250, 1420),
                  color=yellow, fontsize=12, fontweight='bold',
                  bbox={'facecolor':'#162332', 'alpha':.85, 'edgecolor':'none', 'pad':6},
                  arrowprops={'arrowstyle':'->', 'color':yellow, 'lw':1.7})
overview.set_title('1차 기준 사진 center.bmp · 측정 열 x=1316px', fontsize=12)
for j, result in enumerate(results):
    n, mu = result['n'], result['mu_y_px']
    ax = fig.add_subplot(gs[1, j])
    x0, x1, y0, y1 = x-10, x+11, cy-12, cy+13
    ax.imshow(photo[y0:y1, x0:x1], extent=[x0-.5,x1-.5,y1-.5,y0-.5], interpolation='nearest')
    hw = (n-1)//2
    ys = np.arange(cy-hw, cy+hw+1)
    ax.plot([x, x], [y0-.5, y1-.5], ':', color=blue, lw=1)
    ax.scatter(np.full(n,x), ys, s=21, c=blue, edgecolors='white', linewidths=.45, zorder=3)
    ax.scatter([x], [cy], s=90, facecolors='none', edgecolors=yellow, linewidths=2.5, zorder=5)
    ax.annotate('', xy=(x+.7,mu), xytext=(x+8,mu),
                arrowprops={'arrowstyle':'->', 'color':red, 'lw':2.4}, zorder=6)
    ax.plot([x-8,x-8], [cy-hw,cy+hw], color=blue, lw=2.5)
    ax.plot([x-9,x-7], [cy-hw,cy-hw], color=blue, lw=2)
    ax.plot([x-9,x-7], [cy+hw,cy+hw], color=blue, lw=2)
    ax.set_title(f'{n}개 표본 · 위아래 {hw}px', fontsize=12, pad=10)
    ax.set_xlabel(f'계산된 경계 y = {mu:.4f}px', color=red, fontsize=11, labelpad=10)
    ax.set_xticks([])
    ax.set_yticks([1870,1875,1880,1885,1890])
    ax.tick_params(axis='y', labelsize=9)
    ax.set_ylabel('원본 세로 좌표 y [px]' if j==0 else '')
fig.suptitle('같은 사진, 같은 최대점 — 주변을 읽는 범위만 변경', fontsize=17, y=.97)
fig.legend(handles=[Line2D([],[],marker='o',mfc='none',mec=yellow,mew=2.5,ls='',label='노란 원: 최초 최대점 y=1880 (고정)'),
                    Line2D([],[],marker='o',color=blue,ls='',label='파란 점: 사용한 표본 위치'),
                    Line2D([],[],color=red,lw=2,label='빨간 화살표: 계산된 경계 높이')],
           loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False,fontsize=10)
fig.text(.5,.018,'원본 픽셀을 그대로 확대했습니다. 빨간 표시는 한 열의 경계 높이이며, 구조물 전체의 직선 피팅 결과가 아닙니다.',ha='center',fontsize=9,color='#4b5563')
fig.savefig(HERE/'explanation_photo.png')
plt.close(fig)

fig, axes = plt.subplots(1,3,figsize=(12,4.8),sharex=True,sharey=True)
fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.77,wspace=.16)
offsets = np.arange(-10,11)
for ax, result in zip(axes, results):
    n = result['n']; hw = (n-1)//2
    used = abs(offsets)<=hw
    ax.axvspan(-hw,hw,color=blue,alpha=.07)
    ax.plot(offsets,values,'o-',color='#b4b9c0',lw=.8,ms=4)
    ax.scatter(offsets[used],values[used],s=27,color=blue,zorder=3)
    ax.scatter([0],[values[10]],marker='*',s=180,color=yellow,edgecolors='#786500',linewidths=.7,zorder=5)
    u=np.linspace(-hw,hw,401)
    m=result['mu_y_px']-cy
    curve=result['A']*np.exp(-((u-m)**2)/(2*(result['sigma_px']+1e-12)**2))
    ax.plot(u,curve,color=red,lw=2)
    ax.axvline(0,color='#a69500',ls=':',lw=1.2)
    ax.axvline(m,color=red,ls='--',lw=1.2)
    ax.plot(m,result['A'],'o',color=red,ms=5)
    ax.set_title(f'{n}개로 곡선을 맞추면\n중심 y={result["mu_y_px"]:.4f}px',fontsize=12,color=red)
    ax.set_xlim(-10.7,10.7); ax.set_ylim(-15,42)
    ax.set_xticks([-10,-5,0,5,10]); ax.grid(alpha=.15)
    ax.set_xlabel('최초 최대점에서 떨어진 거리 [px]')
axes[0].set_ylabel('세로 밝기 변화량 (Sobel 응답)')
fig.suptitle('최대 표본은 그대로인데, 맞춰진 곡선의 중심은 달라집니다',fontsize=16,y=.96)
fig.text(.5,.86,'사진의 같은 세로 열에서 얻은 실제 표본입니다. 가로축은 사진의 가로 방향이 아니라 위아래 위치입니다.',ha='center',fontsize=10)
fig.legend(handles=[Line2D([],[],marker='*',mfc=yellow,mec='#786500',ls='',ms=11,label='최초 최대점 (고정)'),
                    Line2D([],[],marker='o',color=blue,ls='',label='이번 계산에 사용한 표본'),
                    Line2D([],[],color=red,lw=2,label='저장된 가우시안 피팅 곡선')],
           loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,frameon=False,fontsize=10)
fig.text(.5,.035,'푸른 범위 밖의 회색 점은 이번 계산에 사용하지 않습니다. 원본 밝기값이 아니라 밝기 변화량을 피팅합니다.',ha='center',fontsize=10,color='#4b5563')
fig.savefig(HERE/'explanation_profiles.png')
plt.close(fig)

(HERE/'photo_explanation_provenance.json').write_text(json.dumps({
    'purpose':'Illustration using existing measurements only; no fitting or statistical tests rerun.',
    'source_image':str(source),'source_sha256':ref['sha256'],
    'ref':'R01','target':'lower_structure','x':x,'coarse_y':cy,
    'example_selection':example['selection'],
    'displayed_windows':[5,11,13],
    'note':'Illustrative window comparison at the previously selected representative location, not a maximum-error case.',
    'results':results,
    'rendering':'Matplotlib scientific coordinate overlays; nearest-neighbor photo enlargement; no enhancement.'
},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results))

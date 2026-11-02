"""Illustrate saved line measurements; vertical distances enlarged 40x for display."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

manifest = json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
measurements = json.loads((SOURCE/'measurements.json').read_text(encoding='utf-8'))
images = [next(im for im in manifest if im['dataset']==2 and im['y_mm']==0.05
               and im['theta_deg']==theta and im['repeat']==1) for theta in [0.01,2.0]]
index = {(r['id'],r['target']): r for r in measurements}
before_id, after_id = [im['id'] for im in images]
enlargement = 40
x = np.array([24.0,3816.0])
x_center = x.mean()
fig, axes = plt.subplots(2,1,figsize=(13,12.7))
fig.subplots_adjust(top=.785,bottom=.085,left=.045,right=.97,hspace=.32)
fig.suptitle('기기의 θ가 증가하면, 두 직선은 이렇게 변합니다',fontsize=22,y=.974,fontweight='bold')
fig.text(.5,.926,'실제 2차 측정값 · Y = 0.05mm 고정 · θ = 0.01° → 2.00°',ha='center',fontsize=15)
fig.text(.5,.886,'회색 점선: θ = 0.01°     /     색 실선: θ = 2.00°     /     초록 화살표: 변화 방향',ha='center',fontsize=13,color='#4a5568')
records=[]
for ax,target,color,name,desc in zip(axes,['upper_structure','lower_structure'],
                                    ['#e04743','#138aca'],['상단 직선','하단 직선'],
                                    ['더 완만해짐 · 반시계 방향','더 가팔라짐 · 시계 방향']):
    before=index[(before_id,target)];after=index[(after_id,target)]
    center_y=before['intercept_px']+before['slope']*x_center
    def display_y(row,xx):
        return enlargement*(row['intercept_px']+row['slope']*xx-center_y)
    ax.set_xlim(-180,4020);ax.set_ylim(940,-940);ax.set_aspect('equal')
    ax.set_facecolor('#f8fafc')
    for spine in ax.spines.values():spine.set_color('#cbd5e1');spine.set_linewidth(1.2)
    ax.set_xticks([]);ax.set_yticks([])
    ax.plot(x,[0,0],color='#cbd5e1',lw=1,ls=':',zorder=1)
    ax.plot(x,display_y(before,x),color='#929aa5',lw=3,ls=(0,(7,5)),zorder=2)
    ax.plot(x,display_y(after,x),color=color,lw=3.7,zorder=3)
    # Each arrow follows the difference of the two measured lines at the same x.
    for arrow_x in [420.,3420.]:
        y0=float(display_y(before,arrow_x));y1=float(display_y(after,arrow_x))
        arrow=FancyArrowPatch((arrow_x,y0),(arrow_x,y1),arrowstyle='-|>',
                              mutation_scale=21,lw=2.5,color='#27a45a',
                              shrinkA=7,shrinkB=7,zorder=5)
        ax.add_patch(arrow)
    title=f'{name}   |   {desc}'
    angles=f'영상 각도  {before["image_angle_deg"]:.6f}°  →  {after["image_angle_deg"]:.6f}°'
    ax.set_title(title+'\n'+angles,fontsize=16,loc='left',pad=14,color='#233248',linespacing=1.6)
    ax.text(.025,.035,'영상의 왼쪽',transform=ax.transAxes,fontsize=11,color='#64748b')
    ax.text(.975,.035,'영상의 오른쪽',transform=ax.transAxes,ha='right',fontsize=11,color='#64748b')
    records.append({'target':target,'before':before,'after':after,
                    'display_origin_y_px':center_y})
fig.text(.5,.039,'작은 차이가 보이도록 세로 변화만 40배 확대했습니다. 표시된 각도 숫자는 실제 측정값입니다.',ha='center',fontsize=12,color='#4a5568')
fig.text(.5,.017,'두 선 모두 오른쪽으로 내려갑니다. 상단은 덜 내려가고, 하단은 더 내려가도록 변합니다.',ha='center',fontsize=12,color='#4a5568')
fig.savefig(HERE/'theta_line_change.png',dpi=150,facecolor='white')
fig.savefig(HERE/'theta_line_change.svg',facecolor='white')
plt.close(fig)
provenance={'source_measurements':str(SOURCE/'measurements.json'),
            'sha256':hashlib.sha256((SOURCE/'measurements.json').read_bytes()).hexdigest(),
            'source_images':images,'vertical_display_enlargement':enlargement,
            'coordinate_transform':'x unchanged; displayed y = 40*(measured m*x+b - target-specific before-line midpoint y). Same transform for before and after.',
            'new_image_analysis':False,'fitting_or_filtering_added':False,'measurements':records}
(HERE/'figure_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved theta_line_change.png and theta_line_change.svg')

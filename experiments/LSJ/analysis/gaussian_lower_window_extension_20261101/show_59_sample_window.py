"""Scientific coordinate overlay of the 59-sample vertical window; no new fit."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

HERE=Path(__file__).resolve().parent
info=json.loads((HERE.parent/'gaussian_window_sensitivity_20261101/photo_explanation_provenance.json').read_text(encoding='utf-8'))
source=Path(info['source_image'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==info['source_sha256']
photo=np.array(Image.open(source).convert('RGB'))
assert photo.shape[:2]==(2160,3840)
x,cy=info['x'],info['coarse_y']
n=59; half=(n-1)//2; ya,yb=cy-half,cy+half
assert (ya,yb)==(1851,1909)
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,
                     'font.size':11,'savefig.dpi':150})
fig,axes=plt.subplots(1,2,figsize=(14,6.5),gridspec_kw={'width_ratios':[2,1]},facecolor='white')
fig.subplots_adjust(left=.025,right=.97,bottom=.15,top=.81,wspace=.13)
red='#e52c48'; cyan='#0086bf'; yellow='#ffdc00'
outline=[pe.Stroke(linewidth=4,foreground='white'),pe.Normal()]
x0,x1=x-60,x+75; y0,y1=cy-50,cy+51
ax=axes[0]; ax.imshow(photo); ax.set_axis_off()
ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,ec=cyan,lw=1.5,ls='--'))
ax.plot([x,x],[ya,yb],color=red,lw=2.4,path_effects=outline)
ax.plot([x-12,x+12],[ya,ya],color=red,lw=1.5)
ax.plot([x-12,x+12],[yb,yb],color=red,lw=1.5)
ax.annotate('빨간 짧은 선이 59개 표본 범위',xy=(x,cy),xytext=(200,1600),
            fontsize=12,color='white',bbox={'facecolor':'#263445','edgecolor':'none','pad':7},
            arrowprops={'arrowstyle':'->','color':red,'lw':2})
ax.set_title('전체 사진 · 원본 3840 × 2160px',fontsize=13,pad=10)

ax=axes[1]
ax.imshow(photo[y0:y1,x0:x1],extent=[x0-.5,x1-.5,y1-.5,y0-.5],interpolation='nearest')
ys=np.arange(ya,yb+1)
ax.plot([x,x],[ya,yb],color=red,lw=1,alpha=.55)
ax.scatter(np.full(n,x),ys,s=6,color=red,zorder=3)
ax.scatter([x],[cy],s=80,facecolors='none',edgecolors=yellow,linewidths=2,zorder=4)
ax.plot([x-8,x+36],[ya,ya],color=red,lw=1)
ax.plot([x-8,x+36],[yb,yb],color=red,lw=1)
ax.annotate('',xy=(x+29,ya),xytext=(x+29,yb),
            arrowprops={'arrowstyle':'<->','color':red,'lw':2})
ax.text(x+34,cy,'58px\n중심 간 거리',fontsize=10,color=red,va='center',ha='left',
        bbox={'facecolor':'white','alpha':.88,'edgecolor':'none','pad':3})
ax.text(x-12,cy,'기준점\ny=1880',fontsize=10,va='center',ha='right',color='#263445',
        bbox={'facecolor':'white','alpha':.88,'edgecolor':'none','pad':3})
ax.text(x-12,ya,'1851',fontsize=10,va='center',ha='right',color=red)
ax.text(x-12,yb,'1909',fontsize=10,va='center',ha='right',color=red)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values(): spine.set_edgecolor(cyan); spine.set_linewidth(1.5)
ax.set_title('점선 사각형을 확대 · 빨간 점 59개',fontsize=12,pad=12)
fig.suptitle('하단에서 읽는 세로 범위: 중심 위 29px + 중심 1개 + 아래 29px',fontsize=18,y=.96)
fig.text(.5,.865,'사진의 가로 길이가 아니라, 경계를 가로질러 위아래로 읽는 범위입니다.',ha='center',fontsize=13,color='#3a4d61')
fig.text(.5,.095,'1851~1909번 픽셀에서 1px 간격으로 읽습니다. 표본은 59개, 양 끝 표본 중심 간 거리는 58px입니다.',ha='center',fontsize=12)
fig.text(.5,.045,'원본 사진 세로 높이의 약 2.7% · 실제 사진을 확대해 좌표만 표시했으며, 새 경계 추정은 수행하지 않았습니다.',ha='center',fontsize=10,color='#566274')
out=HERE/'lower_59_sample_window.png'
fig.savefig(out); plt.close(fig)
record={'source':str(source),'source_sha256':info['source_sha256'],'original_dimensions_px':[3840,2160],
        'measurement_column_x':x,'coarse_y':cy,'sample_count':n,'sample_step_px':1,
        'first_sample_center_y':ya,'last_sample_center_y':yb,'center_to_center_span_px':yb-ya,
        'pixel_row_footprint_px':n,'half_width_px':half,
        'source_crop_xyxy':[x0,y0,x1,y1],'original_data_modified':False,'new_fitting':False,
        'figure_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(HERE/'lower_59_sample_window_provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(str(out))

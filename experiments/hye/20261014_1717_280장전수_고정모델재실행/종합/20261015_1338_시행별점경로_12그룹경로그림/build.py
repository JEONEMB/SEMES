from pathlib import Path
import json, hashlib, zipfile
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image, ImageOps, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
RAW=Path(r'C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set')
SOURCE=BASE/'measurement/gn17_spread14.npz'
before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
z=np.load(SOURCE); names=z['names'].astype(str); points=z['points'].astype(float); D=z['D'].astype(float); regions=z['regions']
manifest=pd.read_csv(BASE/'manifest.csv').set_index('file')
plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':12})
COLORS=['#E63946','#F49D18','#B455C7','#2878D0','#00A087','#D95F02','#8C6BB1','#C51B7D','#0086A8','#6B8E23','#BD692E','#4A55B8']
# Stable identities: three QRs, four quadrants per QR. Independent of trial/reference.
groups=np.zeros(len(points),int)
for qr in range(3):
    s=regions==qr; mid=(points[s].min(0)+points[s].max(0))/2
    groups[s]=4*qr+(points[s,0]>=mid[0]).astype(int)+2*(points[s,1]>=mid[1]).astype(int)
assert len(np.unique(groups))==12
pd.DataFrame({'point_id':np.arange(len(points)),'qr':regions+1,'group':groups+1,'x':points[:,0],'y':points[:,1],'color':[COLORS[g] for g in groups]}).to_csv(OUT/'point_groups.csv',index=False,encoding='utf-8-sig')
configs=[('Y2-','Y -0.05',-.05),('Y1-','Y -0.10',-.1),('Y2+','Y +0.05',.05),('Y1+','Y +0.10',.1),('T+','T +0.01',.01),('T-','T +0.01',.01)]
records=[]; allrows=[]
BG='#F8FAFD'; FG='#17324D'
for number,(session,ref_label,rv) in enumerate(configs,1):
    idx=np.array([i for i,n in enumerate(names) if n!='center.bmp' and manifest.loc[n,'session']==session])
    idx=idx[np.argsort([abs(float(manifest.loc[names[i],'value'])) for i in idx])]
    axis='T' if session.startswith('T') else 'Y'
    candidates=[i for i,n in enumerate(names) if n!='center.bmp' and n.startswith(axis) and abs(float(manifest.loc[n,'value'])-rv)<1e-9]
    assert len(candidates)==1
    ref=candidates[0]; cross=ref not in idx
    order=np.r_[ref,idx] if cross else idx
    p0=points+D[ref]; delta=D[order]-D[ref]; paths=p0[None]+delta
    assert np.max(np.abs(delta[0]))==0
    assert len(idx)==(100 if axis=='T' else 20)
    fname=names[ref]; rawpath=RAW/('Theta' if axis=='T' else 'Y')/fname
    im=cv2.imdecode(np.fromfile(rawpath,np.uint8),0); assert im is not None
    x0,x1,y0,y1=1425,2375,940,1200
    crop=im[y0:y1,x0:x1]
    residual=delta-np.median(delta,axis=1,keepdims=True)
    fig=plt.figure(figsize=(16,9),facecolor=BG)
    ax=fig.add_axes([.045,.47,.91,.405])
    ax.imshow(crop,cmap='gray',extent=[x0,x1,y1,y0],vmin=0,vmax=300)
    for g in range(12):
        s=np.where(groups==g)[0]
        segments=np.transpose(paths[:,s],(1,0,2))
        ax.add_collection(LineCollection(segments,colors=COLORS[g],linewidths=.85,alpha=.75))
        ax.scatter(p0[s,0],p0[s,1],s=12,facecolors='white',edgecolors=COLORS[g],linewidths=.8,zorder=4)
        ax.scatter(paths[-1,s,0],paths[-1,s,1],s=10,c=COLORS[g],marker='>',zorder=5)
    ax.set(xlim=(x0,x1),ylim=(y1,y0)); ax.set_aspect('equal'); ax.set_axis_off()
    ax.set_title('실제 사진 위 전체 특징점 경로 · 원 = 기준 위치 / 삼각형 = 마지막 위치 · 가로·세로 1배',loc='left',fontsize=13,color=FG,pad=9)
    # Residual motion shows weak spatial deformation without hiding common motion above.
    for qr in range(3):
        ax=fig.add_axes([.05+qr*.315,.13,.275,.255])
        mask=regions==qr
        lo=p0[mask].min(0)-22; hi=p0[mask].max(0)+22
        ax.imshow(im,cmap='gray',vmin=-100,vmax=400)
        for g in range(4*qr,4*qr+4):
            s=np.where(groups==g)[0]; pp=p0[s][None]+40*residual[:,s]
            ax.add_collection(LineCollection(np.transpose(pp,(1,0,2)),colors=COLORS[g],linewidths=.8,alpha=.8))
            ax.scatter(p0[s,0],p0[s,1],s=12,facecolors='white',edgecolors=COLORS[g],linewidths=.7)
            ax.scatter(pp[-1,:,0],pp[-1,:,1],s=8,c=COLORS[g],marker='>')
        ax.set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]));ax.set_aspect('equal');ax.set_axis_off()
        ax.set_title(f'QR {qr+1} · 그룹 {qr*4+1:02d}–{qr*4+4:02d}',fontsize=12,color=FG,pad=5)
    end=np.median(delta[-1],axis=0)
    fig.text(.045,.952,f'Trial {number:02d}  |  {session}  특징점 이동 경로',fontsize=25,color=FG,weight='bold')
    fig.text(.045,.913,f'기준 {ref_label}  →  {names[idx[-1]][:-4]}   |   trial 사진 {len(idx)}장 · QR 특징점 {len(points)}개 · 12색 그룹',fontsize=14,color='#526779')
    fig.text(.05,.414,'미세 변형 보기: 각 사진에서 전체 점의 중앙값 이동을 뺀 경로 ×40  (위 그림은 실제 크기)',fontsize=13,color=FG,weight='bold')
    handles=[Line2D([0],[0],color=COLORS[g],lw=3,label=f'{g+1:02d}') for g in range(12)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=12,frameon=False,handlelength=1.1,columnspacing=1.15,fontsize=11)
    foot='각 QR의 좌상·우상·좌하·우하 순으로 4개 그룹. 모든 그림에서 같은 점은 같은 색. 선은 측정값 연결이며 회귀 곡선이 아닙니다.'
    fig.text(.045,.043,foot,fontsize=10,color='#526779')
    extra='T−는 요청대로 다른 trial의 T+0.01을 기준으로 사용. 첫 연결은 trial 간 연결입니다.' if cross else '기준 사진을 포함해 |명령값| 증가 순으로 연결. 시간 순서·로봇의 연속 이동 궤적을 의미하지 않습니다.'
    fig.text(.045,.022,extra+f'  끝점 중앙값 Δx={end[0]:+.3f}px, Δy={end[1]:+.3f}px',fontsize=9,color='#526779')
    stem=f'{number:02d}_{session.replace("+","plus").replace("-","minus")}_12그룹_특징점경로'
    fig.savefig(OUT/(stem+'.png'),dpi=200,facecolor=BG)
    fig.savefig(OUT/(stem+'.svg'),facecolor=BG);plt.close(fig)
    records.append({'trial':session,'reference':fname,'cross_trial_reference':bool(cross),'trial_photos':len(idx),'plotted_frames':len(order),'points':len(points),'groups':12,'png':stem+'.png','last_photo':names[idx[-1]],'end_median_dx':end[0],'end_median_dy':end[1]})
    for k,i in enumerate(order):
        for j in range(len(points)):
            allrows.append((session,names[i],j,int(groups[j])+1,*paths[k,j],*delta[k,j]))
    print(session,fname,len(idx),flush=True)
pd.DataFrame(allrows,columns=['trial','file','point_id','group','x','y','dx_from_reference','dy_from_reference']).to_csv(OUT/'trajectories.csv',index=False,encoding='utf-8-sig')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before
(OUT/'verification.json').write_text(json.dumps({'source_sha256_before_and_after':before,'source':'gn17_spread14.npz','measurement':'Existing center-anchored GN measurements, re-referenced by D(photo)-D(reference); no new image tracking or model fitting.','center_plotted':False,'grouping':'QR x four spatial quadrants; not P12 model coefficients','trials':records},ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'읽어주세요.txt').write_text('발표용 6개 PNG: 3200×1800. SVG는 벡터 확대용입니다.\n원=기준 위치, 삼각형=마지막 위치. 위: 실제 크기 전체 경로. 아래: 전체 중앙값 이동 제거 후 ×40.\n12그룹은 QR 3개×공간 4분면이며 P12의 12계수와 다릅니다.\n기존 GN 특징점 좌표를 기준 사진 좌표로 재표현했습니다. 새 기준 영상으로 추적을 다시 수행한 결과는 아닙니다. center는 경로에 넣지 않았으나 원래 측정의 공통 기준으로는 사용되었습니다.\n요청에 T+0.01이 두 번 있어 T−도 T+0.01을 기준으로 표시했습니다. T− 첫 연결은 다른 trial과의 연결입니다.\n사진은 |명령값| 증가 순으로 연결했으므로 실제 시간 순 이동 경로는 아닙니다.\n기존 모델·결과는 변경하지 않았습니다.\n',encoding='utf-8')
thumb=Image.new('RGB',(1600,1350),'white')
for k,r in enumerate(records):
    a=Image.open(OUT/r['png']).convert('RGB');a.thumbnail((800,450));thumb.paste(a,((k%2)*800,(k//2)*450))
thumb.save(OUT/'6개_trial_미리보기.jpg',quality=94)
html='<meta charset="utf-8"><title>6개 trial 특징점 경로</title><style>body{margin:24px;background:#edf1f6;font-family:Malgun Gothic}img{width:100%;max-width:1600px;display:block;margin:12px 0 35px}a{color:#17324d}</style><h1>6개 trial · 12색 특징점 경로</h1><p>원: 기준 / 삼각형: 마지막 위치. 위 실제 크기, 아래 중앙값 이동 제거 ×40. 그룹은 QR별 공간 4분면입니다.</p>'
for r in records: html+=f'<h2>{r["trial"]} · 기준 {r["reference"]}</h2><a href="{r["png"]}">원본 PNG 열기</a><img src="{r["png"]}">'
(OUT/'6개_trial_특징점경로.html').write_text(html,encoding='utf-8')
with zipfile.ZipFile(OUT/'발표용_6개_trial_특징점경로.zip','w',zipfile.ZIP_DEFLATED) as zz:
    for p in OUT.iterdir():
        if p.suffix in ('.png','.svg','.txt','.json','.html','.jpg'):zz.write(p,p.name)
print('done',OUT,flush=True)

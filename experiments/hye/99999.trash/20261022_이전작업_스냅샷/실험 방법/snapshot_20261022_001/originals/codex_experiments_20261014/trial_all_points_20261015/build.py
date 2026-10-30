from pathlib import Path
import sys,json,hashlib,zipfile,time
import numpy as np,pandas as pd,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
OUT=Path(__file__).resolve().parent; BASE=OUT.parent
sys.path.insert(0,str(BASE/'measurement'))
from spread import SpreadPairExtractor
from measure import ROOT,CROP,BOXES
plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':12})
COLORS=['#E63946','#F49D18','#B455C7','#2878D0','#00A087','#D95F02','#8C6BB1','#C51B7D','#0086A8','#6B8E23','#BD692E','#4A55B8']
BG='#17191F';FG='#F4F5F7';SUB='#C3C8D2';GRID='#383D47'
configs=[('Y2-',-.05),('Y1-',-.10),('Y2+',.05),('Y1+',.10),('T+',.01),('T-',-.01)]
manifest=pd.read_csv(BASE/'manifest.csv'); records=[]
def raw(name):return ROOT/('Theta' if name.startswith('T') else 'Y')/name
def style(ax):
    ax.set_facecolor(BG);ax.tick_params(colors=SUB,labelsize=10)
    for sp in ax.spines.values():sp.set_color(GRID)
    ax.grid(color=GRID,alpha=.6)
for no,(session,rv) in enumerate(configs,1):
    table=manifest[manifest.session==session].copy(); table=table.iloc[np.argsort(np.abs(table.value.to_numpy()))]
    names=table.file.to_numpy(dtype=str); vals=table.value.to_numpy(); assert abs(vals[0]-rv)<1e-9
    refname=names[0]; cache=BASE/'trial_direct_reference_20261015'/(session.replace('+','plus').replace('-','minus')+'_direct_tracks.npz')
    if not cache.exists():
        # The reference image is supplied directly. No old feature or displacement file is loaded.
        model=SpreadPairExtractor(raw(refname),features=None,min_distance=14)
        pts=model.points+np.array(CROP[:2]); reg=model.regions
        ds=[np.zeros_like(pts,dtype=np.float32)]; vs=[np.ones(len(pts),bool)]; qs=[np.zeros((len(pts),5),np.float32)]
        for k,n in enumerate(names[1:],1):
            r=model.extract(raw(n));ds.append(r['D']);vs.append(r['valid']);qs.append(r['quality'])
            if k%20==0:print('tracking',session,k+1,'/',len(names),flush=True)
        np.savez_compressed(cache,names=names,values=vals,points=pts,regions=reg,D=np.stack(ds),valid=np.stack(vs),quality=np.stack(qs),reference=np.array(refname))
    z=np.load(cache); pts=z['points'];reg=z['regions'];d=z['D'].astype(float); valid=z['valid'];quality=z['quality'];assert np.all(d[0]==0)
    groups=np.zeros(len(pts),int)
    for q,(x0,y0,x1,y1) in enumerate(BOXES):
        s=reg==q;groups[s]=4*q+(pts[s,0]>=(x0+x1)/2).astype(int)+2*(pts[s,1]>=(y0+y1)/2).astype(int)
    good=valid.all(0);assert good.any();assert len(np.unique(groups[good]))==12
    median=np.median(d[:,good],axis=1);res=d-median[:,None]
    # Representative tracks: each group must appear; prefer low fit residual and spaced origins.
    fit=np.linalg.lstsq(np.vander(vals,3),d.reshape(len(vals),-1),rcond=None)[0]
    fitted=(np.vander(vals,3)@fit).reshape(d.shape)
    noise=np.std(d-fitted,axis=0)[:,1]
    selected=[]
    for g in range(12):
        cand=np.where((groups==g)&good)[0]
        score=noise[cand]+.002*np.mean(quality[:,cand,0],axis=0)
        if selected:
            spacing=np.min(np.abs(pts[cand,None,1]-pts[np.array(selected),1]),axis=1)
            score+=.015*np.maximum(0,15-spacing)
        selected.append(int(cand[np.argmin(score)]))
    # A second representative in each group, if it does not crowd a nearby trajectory.
    length=abs(median[-1,0])
    for g in range(12):
        cand=np.where((groups==g)&good)[0];cand=cand[np.argsort(noise[cand])]
        for j in cand:
            if all(abs(pts[j,1]-pts[k,1])>15 or abs(pts[j,0]-pts[k,0])>length+18 for k in selected):selected.append(int(j));break
    selected=list(range(len(pts)))
    assert valid.all(), 'Do not silently hide invalid tracks'
    im=cv2.imdecode(np.fromfile(raw(refname),np.uint8),0)
    fig=plt.figure(figsize=(16,9),facecolor=BG)
    fig.text(.045,.955,f'Trial {no:02d}   {session}   |   기준 {refname[:-4]}',fontsize=27,color=FG,weight='bold')
    fig.text(.045,.913,f'각 기준 사진에서 직접 추적한 전체 {len(pts)}점 · 나머지 {len(names)-1}장 경로 모두 표시 · 생략 0점 · 12개 그룹 색상 유지',fontsize=14,color=SUB)
    ax=fig.add_axes([.045,.475,.91,.39]);style(ax)
    xa,xb,ya,yb=1400,2390,900,1235
    ax.imshow(im[ya:yb,xa:xb],cmap='gray',extent=[xa,xb,yb,ya],vmin=-45,vmax=350)
    for j in selected:
        color=COLORS[groups[j]]
        smooth=pts[j]+(fitted[:,j]-fitted[0,j])*[1,100]
        measured=pts[j]+d[:,j]*[1,100]
        ax.plot(smooth[:,0],smooth[:,1],color='black',lw=2.1,alpha=.35,zorder=2)
        ax.plot(smooth[:,0],smooth[:,1],color=color,lw=1.05,alpha=.85,zorder=3)
        ax.scatter(measured[:,0],measured[:,1],s=2,color=color,alpha=.28,edgecolors='none',zorder=4)
        ax.scatter(*pts[j],s=11,c='white',edgecolors='#17191F',lw=.45,zorder=6)
        end=smooth[-1];start=smooth[-max(2,len(smooth)//8)]
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',color=color,lw=1.0,mutation_scale=7),zorder=5)
    ax.set(xlim=(xa,xb),ylim=(yb,ya));ax.set_aspect('equal');ax.set_axis_off()
    ax.set_title('전체 특징점: 흰 점 = 기준 위치 / 선 = 2차 추세선 / 옅은 점 = 실제 측정  |  가로 ×1 · 세로 ×100',fontsize=12,color=SUB,loc='left',pad=8)
    ax=fig.add_axes([.065,.14,.30,.25]);style(ax)
    ax.plot(vals,median[:,0],color='#E4E9F1',lw=2.5)
    ax.scatter(vals,median[:,0],c=np.arange(len(vals)),cmap='plasma',s=13,zorder=3)
    ax.set_xlabel('명령값'+(' Y (mm)' if session[0]=='Y' else ' T'),color=SUB)
    ax.set_ylabel('기준 대비 중앙값 Δx (px)',color=SUB)
    ax.set_title('실제 측정: 명령값에 따른 공통 이동',loc='left',color=FG,fontsize=13,pad=10)
    ax.text(.04,.08,f'끝값 Δx {median[-1,0]:+.3f} px\n끝값 Δy {median[-1,1]:+.3f} px',transform=ax.transAxes,color=FG,fontsize=11,bbox=dict(facecolor=BG,edgecolor='none',alpha=.85))
    ax=fig.add_axes([.405,.14,.55,.25]);style(ax)
    ax.imshow(im[960:1185,1510:2295],cmap='gray',extent=[1510,2295,1185,960],vmin=-80,vmax=390)
    for g in range(12):
        s=(groups==g)&valid[-1]
        ax.quiver(pts[s,0],pts[s,1],res[-1,s,0],res[-1,s,1],color=COLORS[g],angles='xy',scale_units='xy',scale=1/50,width=.0035,headwidth=3.8,headlength=4.5)
    ax.set(xlim=(1510,2295),ylim=(1185,960));ax.set_aspect('equal');ax.set_axis_off()
    ax.set_title('끝값 변위: 기준 사진 위 미세 변형 ×50 · 공통 이동 제거',loc='left',color=FG,fontsize=13,pad=10)
    handles=[Line2D([0],[0],color=COLORS[g],lw=4,label=f'{g+1:02d}') for g in range(12)]
    fig.legend(handles=handles,ncol=12,loc='lower center',bbox_to_anchor=(.5,.060),frameon=False,labelcolor=FG,handlelength=1.2,columnspacing=1.15,fontsize=12)
    fig.text(.045,.038,'QR 1: 01–04 / QR 2: 05–08 / QR 3: 09–12 · 각 QR에서 좌상·우상·좌하·우하 순. 색상은 기존 그림과 같습니다.',fontsize=10,color=SUB)
    fig.text(.045,.018,f'center 영상·기존 특징점 좌표 미사용. |명령값| 증가 순 연결. 전체 추적 유효율 {100*valid.mean():.2f}% / 전 구간 유효점 {good.sum()}/{len(pts)}. 확대 그림은 물리적 회전각을 나타내지 않습니다.',fontsize=9,color=SUB)
    stem=f'{no:02d}_{session.replace("+","plus").replace("-","minus")}_전체특징점_발표용'
    fig.savefig(OUT/(stem+'.png'),dpi=200,facecolor=BG)
    fig.savefig(OUT/(stem+'.svg'),facecolor=BG);plt.close(fig)
    pd.DataFrame({'point_id':np.arange(len(pts)),'x':pts[:,0],'y':pts[:,1],'group':groups+1,'color':[COLORS[g] for g in groups],'all_frames_valid':good,'representative':np.isin(np.arange(len(pts)),selected)}).to_csv(OUT/(stem+'_점목록.csv'),index=False,encoding='utf-8-sig')
    records.append(dict(trial=session,reference=refname,photos=len(names),points=len(pts),all_frames_valid=int(good.sum()),valid_fraction=float(valid.mean()),representatives=len(selected),groups_counts=np.bincount(groups,minlength=12).tolist(),png=stem+'.png',raw_reference_sha256=hashlib.sha256(raw(refname).read_bytes()).hexdigest()))
    print('rendered',session,len(pts),'valid',valid.mean(),flush=True)
(OUT/'검증기록.json').write_text(json.dumps(dict(method='Fresh Shi-Tomasi detection separately in six non-center reference photos; whole-QR NCC, LK and GN17 sigma .8 direct pair tracking. No center image or prior feature/track NPZ is loaded.',group_identity='Same spatial quadrant rules and colors, not same feature IDs because features were redetected independently.',display='x1 horizontal, x100 vertical: quadratic trend constrained by subtracting fit at reference, with raw measured dots; bottom residual x50.',trials=records),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'설명.txt').write_text('6개 기준 사진에서 새로 검출하고 직접 추적했던 NPZ를 읽어 전체 특징점을 빠짐없이 다시 그렸습니다. center 기준 좌표는 사용하지 않았습니다.\n모든 점을 표시하며 대표점 선택/제외는 하지 않습니다. 12개 그룹 색상 유지.\n위: 전체점 경로. 가로 1배, 세로 100배. 선은 2차 추세선이며 옅은 점이 실제 측정값입니다. 아래 오른쪽 미세변형 50배.\n기존 측정 및 대표점 그림은 보존했습니다.\n',encoding='utf-8')
thumb=Image.new('RGB',(1600,1350),BG)
for k,r in enumerate(records):
    a=Image.open(OUT/r['png']).convert('RGB');a.thumbnail((800,450));thumb.paste(a,((k%2)*800,(k//2)*450))
thumb.save(OUT/'6개_trial_전체특징점_미리보기.jpg',quality=95)
html='<meta charset="utf-8"><style>body{background:#17191f;color:white;font-family:Malgun Gothic;margin:25px}img{width:100%;max-width:1600px;margin-bottom:30px}a{color:#8dd3ff}</style><h1>6개 trial · 전체 특징점 경로</h1><p>대표점 선택 없이 전체 200~208점을 모두 표시. 가로 ×1 / 세로 ×100. 선은 2차 추세선, 옅은 점은 실제 측정. 하단 전체점 미세 변형 ×50.</p>'
for r in records:html+=f'<h2>{r["trial"]} / 기준 {r["reference"]}</h2><a href="{r["png"]}">PNG 원본</a><br><img src="{r["png"]}">'
(OUT/'6개_trial_전체특징점.html').write_text(html,encoding='utf-8')
with zipfile.ZipFile(OUT/'발표용_6장_전체특징점.zip','w',zipfile.ZIP_DEFLATED) as zz:
    for p in OUT.iterdir():
        if p.suffix in ['.png','.svg','.jpg','.txt','.json','.csv','.html','.py','.npz']:zz.write(p,p.name)
print('COMPLETE',flush=True)

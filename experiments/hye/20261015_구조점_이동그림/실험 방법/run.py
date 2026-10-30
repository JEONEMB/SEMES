from pathlib import Path
import sys,json,time,hashlib,concurrent.futures
BASE=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'deps'),str(BASE/'measurement')]
import numpy as np,pandas as pd,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from measure import ROOT,CROP,BOXES,read,coarse,track
from refine import Refiner
plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12,'svg.fonttype':'none'})
cv2.setNumThreads(1)
src=BASE/'subpixel_grid_corners_20261015'
f=pd.read_csv(src/'confidence_bordered_squares/corners_with_quality.csv').fillna({'reason':''})
pts=f[['x','y']].to_numpy();regions=f.qr.to_numpy()-1;conf=f.quality_score.to_numpy()
ref=read(ROOT/'center.bmp');models=[]
for k,(a,b,c,d) in enumerate(BOXES):
    a-=CROP[0];c-=CROP[0];b-=CROP[1];d-=CROP[1]
    template=ref[b:d,a:c].copy();p=(pts[regions==k]-[a+CROP[0],b+CROP[1]]).astype(np.float32)
    assert (p[:,0]>=9).all() and (p[:,1]>=9).all() and (p[:,0]<c-a-9).all() and (p[:,1]<d-b-9).all()
    models.append((a,b,c,d,cv2.GaussianBlur(template,(0,0),.8),p,Refiner(template,p,17,.8)))
names=np.load(BASE/'center_fixed_20261015/center_once_tracks.npz')['names'].astype(str)
def extract(name):
    if name=='center.bmp':return np.zeros((len(pts),2)),np.ones(len(pts),bool),np.zeros((len(pts),5))
    cur=read(ROOT/('Y' if name[0]=='Y' else 'Theta')/name)
    init,scores=coarse(ref,cur,include_screws=False)
    D=np.zeros((len(pts),2));valid=np.zeros(len(pts),bool);quality=np.zeros((len(pts),5))
    for k,(a,b,c,d,template,p,model) in enumerate(models):
        dx,dy=init[k].astype(int);target=cur[b+dy:d+dy,a+dx:c+dx].copy()
        ld,lq,lv=track(template,cv2.GaussianBlur(target,(0,0),.8),p,np.zeros_like(p),15)
        gd,gq,gv=model.run(target,ld);s=regions==k
        D[s]=gd+init[k];valid[s]=lv&gv;quality[s]=np.c_[gq,lq[:,0]]
    return D,valid,quality
start=time.time();cache=OUT/'tracks.npz'
if not cache.exists():
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(extract,names)):
            results.append(r)
            if i%40==0:print('TRACK',i+1,len(names),'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(cache,names=names,points=pts,regions=regions,D=np.stack([r[0] for r in results]),valid=np.stack([r[1] for r in results]),quality=np.stack([r[2] for r in results]),confidence=conf)
z=np.load(cache);D=z['D'];valid=z['valid'];quality=z['quality'];index={n:i for i,n in enumerate(names)}
failrows=[dict(file=names[i],point_id=int(j),qr=int(regions[j]+1),grid_x=int(f.grid_x.iloc[j]),grid_y=int(f.grid_y.iloc[j]),fb_error=float(quality[i,j,4]),GN_step=float(quality[i,j,1])) for i,j in zip(*np.where(~valid))]
pd.DataFrame(failrows,columns=['file','point_id','qr','grid_x','grid_y','fb_error','GN_step']).to_csv(OUT/'tracking_failures.csv',index=False,encoding='utf-8-sig')
f.assign(point_id=np.arange(len(f))).to_csv(OUT/'fixed_features.csv',index=False,encoding='utf-8-sig')
np.savez_compressed(OUT/'measurements.npz',names=names,X_current=pts[None]+D,D_center_relative=D,valid=valid)
configs=[('Y_minus_005','Y',-1,.05),('Y_plus_005','Y',1,.05),('Y_minus_010','Y',-1,.1),('Y_plus_010','Y',1,.1),('T_minus','T',-1,.01),('T_plus','T',1,.01)]
entries=[]
for slug,axis,sign,first in configs:
    vals=np.arange(first,2.001,.1) if axis=='Y' else np.arange(.01,1.001,.01)
    special=axis=='Y' and first==.05
    if special:vals=np.r_[vals,2.]
    ns=[f'{axis}{"+" if sign>0 else "-"}{v:.2f}.bmp' for v in vals];assert all(n in index for n in ns)
    ii=np.array([index[n] for n in ns]);entries.append(dict(slug=slug,axis=axis,sign=sign,start=sign*first,end=sign*vals[-1],files=ns,indices=ii.tolist(),extra_endpoint=special))
(OUT/'sequences.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
# Representative selection is fixed using center diagnostics only, never motion labels or test tracks.
representatives=[]
for k,(a,b,c,d) in enumerate(BOXES):
    quadrant=(pts[:,0]>=(a+c)/2).astype(int)+2*(pts[:,1]>=(b+d)/2).astype(int)
    for q in range(4):
        candidates=np.flatnonzero((regions==k)&(quadrant==q))
        chosen=[]
        for j in candidates[np.argsort(-conf[candidates])]:
            if all(np.linalg.norm(pts[j]-pts[t])>=20 for t in chosen):chosen.append(int(j))
            if len(chosen)==3:break
        representatives.extend(chosen)
pd.DataFrame(dict(point_id=representatives)).to_csv(OUT/'representative_ids.csv',index=False)
bg=cv2.imdecode(np.fromfile(ROOT/'center.bmp',np.uint8),0)
xa,xb,ya,yb=1430,2380,880,1260
BG='#17191f';FG='#f4f5f7';SUB='#c6cbd3';cmap=plt.get_cmap('plasma')
def plot(e,selected,representative=False,clean=False):
    ii=np.array(e['indices']);n=len(ii);sgn='−' if e['sign']<0 else '+'
    label=f'{e["axis"]} {sgn}{abs(e["start"]):.2f} → {sgn}{abs(e["end"]):.2f}'
    fig=plt.figure(figsize=(18,10),facecolor=BG)
    fig.text(.035,.956,label+'  |  새 QR 구조 모서리 이동',fontsize=25,weight='bold',color=FG)
    fig.text(.035,.914,('대표 '+str(len(selected))+'점: QR별 4분면 × 각 최대 3점, center 품질순·간격 20px' if representative else '전체 755점 · 동일 ID · 각 사진을 center에서 직접 추적')+'  |  명령값 크기순 연결',fontsize=13,color=SUB)
    for row,amp in enumerate([1,100]):
        ax=fig.add_axes([.035,.495-row*.39,.93,.34]);ax.imshow(bg[ya:yb,xa:xb],cmap='gray',extent=[xa,xb,yb,ya],vmin=-25,vmax=320)
        coords=pts[None]+D[ii]*[1,amp]
        for j in selected:
            v=valid[ii,j];segments=np.stack([coords[:-1,j],coords[1:,j]],axis=1)
            ok=v[:-1]&v[1:];colors=cmap(np.linspace(0,1,n-1));colors[:,3]=(.85 if representative else .22)+.12*conf[j]
            if e['extra_endpoint']:
                # 2.00 is outside the .05,.15,...,1.95 family: never imply one same-family continuation.
                if ok[-1]:ax.plot(*segments[-1].T,color='#ffd34e',lw=1.2,ls='--',alpha=.8)
                ok[-1]=False
            ax.add_collection(LineCollection(segments[ok],colors=colors[ok],linewidths=2.2 if representative else .65,zorder=3))
            s=coords[0,j];end=coords[-1,j]
            if v[0]:ax.scatter(*s,s=40 if representative else 7,marker='s',facecolor='#8bfa83',edgecolor='black',linewidth=.7 if representative else .2,zorder=5,alpha=.95 if representative else .5)
            if v[-1]:ax.scatter(*end,s=27 if representative else 7,marker=(3,0,float(np.degrees(np.arctan2(*(end-s)[::-1])))-90),c='#ffe34e',linewidths=0,zorder=5,alpha=.95 if representative else .65)
            if (~v).any():ax.scatter(coords[~v,j,0],coords[~v,j,1],s=15,marker='x',c='#ff655d',linewidth=.7,zorder=6)
        ax.set(xlim=(xa,xb),ylim=(yb,ya));ax.set_aspect('equal');ax.set_axis_off()
        ax.set_title('실제 변위: 가로 ×1 · 세로 ×1' if amp==1 else '변위 확대: 가로 ×1 · 세로 ×100 (길이·기울기는 실제 배율 아님)',loc='left',color=SUB,fontsize=13,pad=8)
    misses=int((~valid[ii][:,selected]).sum())
    note=f'연두 네모: 시작 사진 위치  ·  보라 → 노랑: 명령값 진행  ·  노랑 화살촉: 끝 위치  ·  빨강 ×: 추적 실패 {misses}건'
    fig.text(.035,.072,note,color=SUB,fontsize=12)
    fig.text(.035,.038,'기준 사진: center 고정. 선은 사진별 측정점 연결이며 시간 순 로봇 궤적이나 회귀 추세선이 아닙니다.',color=SUB,fontsize=11)
    if e['extra_endpoint']:fig.text(.035,.009,'0.05 시작 계열은 0.05, 0.15, …, 1.95이며 2.00은 별도 끝점으로 점선 연결했습니다.',color='#ffd34e',fontsize=11)
    suffix='representative' if representative else 'all755'
    fig.savefig(OUT/f'{e["slug"]}_{suffix}.png',dpi=160,facecolor=BG)
    fig.savefig(OUT/f'{e["slug"]}_{suffix}.svg',facecolor=BG)
    plt.close(fig)
for e in entries:
    plot(e,np.arange(len(pts)));plot(e,np.array(representatives),True)
    print('FIGURE',e['slug'],flush=True)
thumb=Image.new('RGB',(1800,1500),BG)
for k,e in enumerate(entries):
    im=Image.open(OUT/f'{e["slug"]}_representative.png').convert('RGB');im.thumbnail((900,500));thumb.paste(im,((k%2)*900,(k//2)*500))
thumb.save(OUT/'six_sequences_preview.jpg',quality=95)
checks=dict(features=len(pts),center_uncertain_features=int((~f.valid).sum()),photos=280,valid_fraction=float(valid[ names!='center.bmp'].mean()),failed_point_observations=len(failrows),photos_with_failed_points=int((~valid).any(axis=1).sum()),representatives=len(representatives),center_zero=bool(np.all(D[names=='center.bmp']==0)),source_features_sha256=hashlib.sha256((src/'refined_corners.csv').read_bytes()).hexdigest(),tracking='center fixed structural IDs, QR-only NCC, LK15, inverse GN17; valid=LK success forward/backward error <0.5px & finite GN correction <1.5px and last step <.02px',failure_display='Red x; invalid-adjacent segments disconnected; original center-uncertain points retained and recorded',sequence_note='Y .05 family ends1.95, requested2.00 appended with dashed line; T ends ±1.00. Filenames determine display ordering only, never tracking.',runtime_seconds=time.time()-start)
(OUT/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
html='<meta charset="utf-8"><title>새 구조 특징점 이동 시각화</title><style>body{font-family:Malgun Gothic;background:#17191f;color:#eee;margin:24px}img{max-width:100%}a{color:#8bfa83}</style><h1>새 QR 구조 특징점 · 6개 구간 이동</h1><p><a href="../../ALGORITHM_RULES.md">공통 규칙</a> · <a href="checks.json">측정 설정·실패 수</a> · <a href="tracking_failures.csv">추적 실패 목록</a> · <a href="sequences.json">사용 사진</a> · <a href="tracks.npz">전체 추적 좌표</a></p>'
html+='<p>전체 755점이 기본 결과입니다. 대표 36점은 center 품질·공간 간격만으로 선택한 설명용 별도 그림입니다. 이전 205점 좌표를 재사용하지 않고 새 서브픽셀 모서리를 직접 추적했습니다. center에서 불확실했던 147점도 유지·기록했습니다. 추적 유효율은 Y/T 축 판정 정확도가 아닙니다.</p>'
for e in entries:
    html+=f'<h2>{e["slug"]}</h2><a href="{e["slug"]}_all755.png">전체 755점 PNG</a> · <a href="{e["slug"]}_all755.svg">SVG</a><img src="{e["slug"]}_all755.png"><details><summary>대표점 확대 설명 그림</summary><img src="{e["slug"]}_representative.png"></details>'
(OUT/'report.html').write_text(html,encoding='utf-8');print(json.dumps(checks),flush=True)


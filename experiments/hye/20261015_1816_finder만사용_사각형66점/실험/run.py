from pathlib import Path
import sys,json,time,hashlib
import concurrent.futures
BASE=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'deps'),str(BASE),str(BASE/'measurement')]
import numpy as np,pandas as pd,cv2
from measure import ROOT,CROP,BOXES,read,track
from refine import Refiner
import zero_experiment as Z
cv2.setNumThreads(1)
ref=read(ROOT/'center.bmp')
_,bw=cv2.threshold(ref,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cont,h=cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
boxes=[]
for i,c in enumerate(cont):
    if h[0,i,2]<0 or h[0,h[0,i,2],2]<0 or not 500<cv2.contourArea(c)<6000:continue
    x,y,w,hh=cv2.boundingRect(c)
    gx,gy=x+CROP[0],y+CROP[1]
    qr=[k for k,(a,b,cc,d) in enumerate(BOXES) if a<gx<cc and b<gy<d]
    if qr:boxes.append((qr[0],x,y,w,hh))
boxes.sort();assert len(boxes)==9
models=[];points=[];regions=[];patterns=[]
for j,(qr,x,y,w,h) in enumerate(boxes):
    # Inset excludes the outside data region. All blur/LK/GN sampling uses only this crop.
    x+=3;y+=3;w-=6;h-=6
    template=ref[y:y+h,x:x+w].copy();sm=cv2.GaussianBlur(template,(0,0),.65)
    mask=np.zeros_like(template);mask[10:-10,10:-10]=255
    p=cv2.goodFeaturesToTrack(sm,100,.02,4,mask=mask,blockSize=5)
    assert p is not None
    p=cv2.cornerSubPix(sm,p,(2,2),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.001)).reshape(-1,2)
    p=p[(p[:,0]>=10)&(p[:,0]<w-10)&(p[:,1]>=10)&(p[:,1]<h-10)]
    assert len(p)>=4
    models.append((x,y,w,h,template,cv2.GaussianBlur(template,(0,0),.8),p,Refiner(template,p,17,.8)))
    points.extend(p+[x+CROP[0],y+CROP[1]]);regions.extend([qr]*len(p));patterns.extend([j]*len(p))
pts=np.array(points);regions=np.array(regions);patterns=np.array(patterns)
canvas=cv2.cvtColor(ref,cv2.COLOR_GRAY2BGR)
for j,(x,y,w,h,*_) in enumerate(models):
    cv2.rectangle(canvas,(x,y),(x+w-1,y+h-1),(0,180,255),1)
    for p in pts[patterns==j]-CROP[:2]:cv2.circle(canvas,tuple(np.rint(p).astype(int)),2,(0,0,255),-1)
    cv2.putText(canvas,str(j+1),(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,180,255),1)
cv2.imencode('.png',canvas)[1].tofile(OUT/'features.png')
pd.DataFrame(dict(point_id=np.arange(len(pts)),x=pts[:,0],y=pts[:,1],qr=regions+1,pattern=patterns+1)).to_csv(OUT/'features.csv',index=False)
old=np.load(BASE/'center_fixed_20261015/center_once_tracks.npz');names=old['names'].astype(str)
def job(name):
    if name=='center.bmp':return np.zeros((len(pts),2)),np.ones(len(pts),bool),np.zeros((len(pts),5))
    cur=read(ROOT/('Y' if name.startswith('Y') else 'Theta')/name);ds=[];vs=[];qs=[]
    for x,y,w,h,t,sm,p,model in models:
        search=cur[y-22:y+h+22,x-110:x+w+110]
        cc=cv2.matchTemplate(search,t,cv2.TM_CCOEFF_NORMED);_,score,_,pos=cv2.minMaxLoc(cc)
        dx,dy=pos[0]-110,pos[1]-22
        target=cur[y+dy:y+h+dy,x+dx:x+w+dx].copy()
        d,lq,lv=track(sm,cv2.GaussianBlur(target,(0,0),.8),p,np.zeros_like(p),15)
        gd,gq,gv=model.run(target,d)
        ds.append(gd+[dx,dy]);vs.append(lv&gv);qs.append(np.c_[gq,lq[:,0]])
    return np.vstack(ds),np.concatenate(vs),np.vstack(qs)
start=time.time();cache=OUT/'tracks.npz'
if not cache.exists():
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(job,names)):
            results.append(r)
            if i%40==0:print('TRACK',i+1,len(names),'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(cache,names=names,points=pts,regions=regions,patterns=patterns,D=np.stack([r[0] for r in results]),valid=np.stack([r[1] for r in results]),quality=np.stack([r[2] for r in results]))
z=np.load(cache);D=z['D'];valid=z['valid']
pd.DataFrame([dict(file=names[i],point_id=int(j),pattern=int(patterns[j]+1)) for i,j in zip(*np.where(~valid))]).to_csv(OUT/'tracking_failures.csv',index=False)
x=(pts[:,0]-1906)/486;y=(pts[:,1]-1070)/122;B=np.array([x*0+1,x,y,x*x,x*y,y*y]).T
X=[];usable=[]
for dd,v in zip(D,valid):
    good=np.linalg.matrix_rank(B[v])==6
    usable.append(good)
    X.append(np.linalg.lstsq(B[v],dd[v],rcond=None)[0].reshape(12) if good else np.full(12,np.nan))
X=np.array(X);assert all(usable),'Unusable photos must be retained as explicit inference failures'
m=Z.manifest(names);idx={n:i for i,n in enumerate(names)};ids=np.flatnonzero(m.axis!='C')
folds=json.loads((BASE/'gls_comparison_20261014/splits.json').read_text(encoding='utf-8'))
folds=[dict(protocol='full280_resubstitution',fold='all',train=names[ids].tolist(),test=names[ids].tolist())]+[f for f in folds if f['protocol']!='full280']
(OUT/'splits.json').write_text(json.dumps(folds,ensure_ascii=False,indent=2),encoding='utf-8')
preds=[]
for quad in [True,False]:
    method='finder_quadratic' if quad else 'finder_linear'
    for ss in folds:
        tr=np.array([idx[n] for n in ss['train']]);te=np.array([idx[n] for n in ss['test']])
        model=Z.fit(m,X,tr,m.value.to_numpy(),'center',quad)
        df=Z.predict(m,X,te,model,'center');df['method']=method;df['protocol']=ss['protocol'];df['fold']=ss['fold'];preds.append(df)
        if ss['protocol']=='full280_resubstitution':
            branches,W=model
            np.savez_compressed(OUT/(method+'_model.npz'),W=W,**{a+'_'+key:v[k] for a,v in branches.items() for key,k in [('base',1),('slope',2),('quadratic',3)]})
p=pd.concat(preds,ignore_index=True)
truth=np.c_[np.where(p.axis=='Y',p.value,0),np.where(p.axis=='T',p.value,0)]
pr=np.c_[np.where(p.pred_axis=='Y',p.pred,0),np.where(p.pred_axis=='T',p.pred,0)]
p['Y_abs_error']=abs(truth-pr)[:,0];p['T_abs_error']=abs(truth-pr)[:,1]
p['pass']=(p.axis==p.pred_axis)&(np.sign(p.value)==np.sign(p.pred))&(p.Y_abs_error<=.09)&(p.T_abs_error<=.09)
p.to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig');p[~p['pass']].to_csv(OUT/'failures.csv',index=False,encoding='utf-8-sig')
metrics=pd.DataFrame([dict(method=a,protocol=b,**Z.score(g)) for (a,b),g in p.groupby(['method','protocol'])])
metrics.to_csv(OUT/'metrics.csv',index=False)
baseline=pd.read_csv(BASE/'center_fixed_20261015/metrics.csv');comp=pd.concat([baseline[baseline.method=='center_quadratic'],metrics])
comp.to_csv(OUT/'comparison.csv',index=False)
meta=dict(features=len(pts),features_per_pattern=np.bincount(patterns).tolist(),pattern_crops=[list(v[:4]) for v in models],valid_rate=float(valid.mean()),invalid_tracks=int((~valid).sum()),unusable_photos=int((~np.array(usable)).sum()),spatial_rank=int(np.linalg.matrix_rank(B)),spatial_condition=float(np.linalg.cond(B)),runtime_seconds=time.time()-start,scope='Existing fixed QR images only; new random payloads and changed size/version not independently validated',tracking='Finder crop only for NCC, LK15 and GN17; crop inset 3px; features at least 10px inside inset crop; center only ShiTomasi selection',failure_policy='Per image valid points only for spatial least squares; never substitute point IDs; rank deficient images must be reported as inference failures',rules_sha256=hashlib.sha256((BASE.parent/'ALGORITHM_RULES.md').read_bytes()).hexdigest())
(OUT/'checks.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
cols=['method','protocol','n','Y_axis_sign_accuracy','T_axis_sign_accuracy','strict_pass09','vector_Y_max','vector_T_max']
html='<meta charset="utf-8"><title>큰 위치 검출 사각형만의 검증</title><style>body{font-family:Malgun Gothic;margin:28px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px}img{max-width:100%}</style><h1>QR 큰 위치 검출 사각형만의 검증</h1><p><a href="../../ALGORITHM_RULES.md">공통 규칙</a> · <a href="predictions.csv">사진별 예측</a> · <a href="failures.csv">실패</a> · <a href="checks.json">측정 설정</a></p>'
html+=f'<p>9개 사각형, center 고정 {len(pts)}점. 전체 QR 무늬 및 나사 제외: 초기 탐색과 정밀 추적 모두 사각형 내부 crop만 사용. 추적 실패 {(~valid).sum()}건, 유효율 {valid.mean():.6%}. 모든 사진 공간식 계산 가능: {all(usable)}.</p><img src="features.png"><p>주 비교는 finder_quadratic과 center_quadratic. linear는 사전 지정 비교이며 결과에 따라 주 방법을 교체하지 않음. P12와 원점 통과 Y/T 곡선 및 학습 잔차 공분산은 기존과 동일. 모든 모델/공분산은 분할별 학습.</p>'
html+=comp[cols].to_html(index=False,float_format=lambda v:f'{v:.6f}')
html+='<p>full280_resubstitution은 재대입, repeated5는 동일 사진의 반복 교차검증. 새 랜덤 QR 촬영에 대한 독립 검증이 아니다. QR 내용 변경 시 일반화는 추가 촬영이 필요하며, 위치/크기/버전 변화와 경계 주변 인쇄 영향을 포함해야 한다. 정답은 파일명 명령값이다.</p>'
(OUT/'report.html').write_text(html,encoding='utf-8')
print(json.dumps(meta),flush=True);print(comp[comp.protocol=='full280_resubstitution'][cols].to_string(index=False),flush=True)

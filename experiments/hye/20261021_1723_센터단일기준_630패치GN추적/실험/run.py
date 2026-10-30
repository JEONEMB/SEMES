from pathlib import Path
import sys,json,time,hashlib,concurrent.futures
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent; BASE=HERE.parent; ROOT=BASE.parent
sys.path[:0]=[str(BASE),str(BASE/'measurement')]
import zero_experiment as Z
from spread import SpreadPairExtractor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def spatial(points,D,weights=None):
    u=(points[:,0]-1906)/486;v=(points[:,1]-1070)/122
    B=np.c_[u*0+1,u,v,u*u,u*v,v*v]
    if weights is None:
        return np.linalg.lstsq(B,D.transpose(1,0,2).reshape(len(points),-1),rcond=None)[0].reshape(6,len(D),2).transpose(1,0,2).reshape(-1,12)
    out=[]
    for d,w in zip(D,weights):
        sw=np.sqrt(w)
        out.append(np.linalg.lstsq(B*sw[:,None],d*sw[:,None],rcond=None)[0].ravel())
    return np.array(out)

def main():
    source=ROOT/'analysis_grid_corners_20261019/out/corners_280.csv'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    c=pd.read_csv(source); ref=c[c.file=='center.bmp'].sort_values('id'); ref=ref[ref.black_neighbors!=2]
    assert len(ref)==630
    names=np.array(sorted(c.file.unique(),key=lambda n:(n!='center.bmp',n)))
    pts=ref[['x','y']].to_numpy(); ids=ref.id.to_numpy()
    # Region inferred geometrically, independent of labels and QR payload names.
    regions=np.argmin(abs(pts[:,0,None]-np.array([1644,1901,2159])),axis=1)
    np.savez(HERE/'features.npz',points=pts,regions=regions,strengths=np.ones(630),ids=ids)
    m=Z.manifest(names); cache=HERE/'tracks.npz'
    if not cache.exists():
        extractor=SpreadPairExtractor(Z.ROOT/'center.bmp',str(HERE/'features.npz'))
        start=time.time(); results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            for i,r in enumerate(ex.map(extractor.extract,[Path(p) for p in [Z.ROOT/('center.bmp' if n=='center.bmp' else ('Y/' if n[0]=='Y' else 'Theta/')+n) for n in names]])):
                results.append(r)
                if i%20==0:print('tracking',i+1,281,round(time.time()-start,1),flush=True)
        np.savez_compressed(cache,names=names,points=pts,ids=ids,D=np.stack([r['D'] for r in results]),quality=np.stack([r['quality'] for r in results]),valid=np.stack([r['valid'] for r in results]),seconds=np.array([r['runtime_seconds'] for r in results]))
    a=np.load(cache);assert np.array_equal(a['names'],names)
    D=a['D'].astype(float);D-=D[0];assert np.isfinite(D).all()
    xp=c.pivot(index='file',columns='id',values='x').loc[names,ids].to_numpy();yp=c.pivot(index='file',columns='id',values='y').loc[names,ids].to_numpy()
    oldD=np.stack([xp-xp[0],yp-yp[0]],axis=-1)
    # Fixed observable quality weighting; no labels or cross-image test statistics.
    q=a['quality'];w=1/(1+(q[:,:,0]/6)**2+(q[:,:,4]/.15)**2)
    w*=np.where(a['valid'],1,.05);w=np.maximum(w,.001)
    configs={'v1_edge630':spatial(pts,oldD),'v2_patch630':spatial(pts,D),'v2_patch630_quality':spatial(pts,D,w)}
    folds=json.loads((BASE/'gls_comparison_20261014/splits.json').read_text());ix={n:i for i,n in enumerate(names)};records=[]
    for key,X in configs.items():
        np.save(HERE/(key+'_features.npy'),X)
        for s in folds:
            tr=np.array([ix[n] for n in s['train']]);te=np.array([ix[n] for n in s['test']])
            model=Z.fit(m,X,tr,m.value.to_numpy(),'center',True)
            records.append(Z.predict(m,X,te,model,'center').assign(method=key,protocol=s['protocol'],fold=s['fold']))
            if s['protocol']=='full280':
                branches,W=model
                (HERE/(key+'_model.json')).write_text(json.dumps(dict(W=W.tolist(),branches={k:[v[0],*[z.tolist() for z in v[1:]]] for k,v in branches.items()},zero_feature=[0]*12),indent=2))
        print('evaluated',key,flush=True)
    p=pd.concat(records,ignore_index=True)
    p['correct']=(p.axis==p.pred_axis)&(np.sign(p.value)==np.sign(p.pred))
    for axis in ['Y','T']:p['e'+axis]=abs(np.where(p.axis==axis,p.value,0)-np.where(p.pred_axis==axis,p.pred,0))
    p['pass']=p.correct&(p.eY<=.09)&(p.eT<=.09)
    p.to_csv(HERE/'predictions.csv',index=False,encoding='utf-8-sig');p[~p['pass']].to_csv(HERE/'failures.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for key,g in p.groupby('method'):
        for protocol,gg in [('full280',g[g.protocol=='full280']),('repeated20',g[g.protocol.str.startswith('repeated5_')]),('interleaved',g[g.protocol=='interleaved'])]:
            rows.append(dict(method=key,protocol=protocol,Y_correct=int(gg[gg.axis=='Y'].correct.sum()),Y_n=int((gg.axis=='Y').sum()),T_correct=int(gg[gg.axis=='T'].correct.sum()),T_n=int((gg.axis=='T').sum()),failures=int((~gg['pass']).sum()),Y_max=gg.eY.max(),T_max=gg.eT.max()))
    summary=pd.DataFrame(rows);summary.to_csv(HERE/'summary.csv',index=False,encoding='utf-8-sig')
    assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
    audit=dict(source_sha256=digest,ids=630,images=280,reference='center.bmp',origins=0,valid_rate=float(a['valid'][1:].mean()),minimum_image_valid_rate=float(a['valid'][1:].mean(1).min()),median_pair_seconds_concurrent=float(np.median(a['seconds'][1:])),support='Each QR crop only; spline nearest-border extension and LK crop border padding for edge points, no external plate pixels',weights='1/(1+(RMSE/6)^2+(FB/.15)^2), invalid x .05, floor .001; fixed before evaluation',evaluation='all saved119 splits, training-only curve and covariance, no test labels in inference; commanded labels not measured physical pose',limitations=['Existing center structural IDs reused; fresh installation QR detection/generalization not tested','Full280 is resubstitution, repeated CV is same acquisition sessions','No 6-origin offsets; zero center feature gives zero output'])
    (HERE/'audit.json').write_text(json.dumps(audit,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    s=summary[summary.protocol=='repeated20'];labels=['Boundary fit','Actual patch','Patch + quality'];x=np.arange(3)
    axs[0].bar(x,s.failures);axs[0].set_title('Failures / 5600 held-out predictions')
    axs[1].bar(x,s.Y_max);axs[1].axhline(.09,color='red',ls='--');axs[1].set_title('Maximum Y error (all cases)')
    axs[2].bar(x,s.T_max);axs[2].axhline(.09,color='red',ls='--');axs[2].set_title('Maximum T error (all cases)')
    for ax in axs:ax.set_xticks(x,labels,rotation=15);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=180);plt.close(fig)
    html='<meta charset="utf-8"><style>body{font:18px sans-serif;max-width:1200px;margin:40px auto}td,th{padding:10px;border:1px solid #ddd}img{width:100%}</style><h1>센터 기준 QR 변위 추정 v2</h1><p>630개 동일 구조 ID → 센터 실제 패치 LK/GN 추적 → 현재−센터 → P12 → 원점 통과 Y/T 이차곡선 GLS 비용 비교. 6개 0점 후보 없음.</p><p>full280: 재대입 / repeated20: 동일 분할 20회 5겹 교차검증(5600개 예측) / interleaved: 기존 교차 분할. 모든 오분류를 두 축 최대오차에 포함.</p>'+summary.to_html(index=False)+'<img src="comparison.png"><p>새 설치 QR 자동 검출과 독립 촬영 일반화, 실시간 카메라 성능은 검증하지 않았습니다. 기존 center 구조 ID와 고정 QR 영역을 재사용한 오프라인 측정 실험입니다.</p>'
    (HERE/'report.html').write_text(html,encoding='utf-8');print(summary.to_string(index=False),flush=True);print(json.dumps(audit),flush=True)
if __name__=='__main__':main()

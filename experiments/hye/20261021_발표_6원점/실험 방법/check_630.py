from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;PROJ=BASE.parent
sys.path.insert(0,str(BASE));import zero_experiment as Z
OUT=HERE/'verification_630';OUT.mkdir(exist_ok=True)
source=PROJ/'analysis_grid_corners_20261019/out/corners_280.csv';before=hashlib.sha256(source.read_bytes()).hexdigest()
C=pd.read_csv(source);names=np.array(sorted(C.file.unique(),key=lambda n:(n!='center.bmp',n)));m=Z.manifest(names);ix={n:i for i,n in enumerate(names)}
xp=C.pivot(index='file',columns='id',values='x').loc[names];yp=C.pivot(index='file',columns='id',values='y').loc[names]
ref=C[C.file=='center.bmp'].set_index('id').loc[xp.columns];sel=ref.black_neighbors.to_numpy()!=2;assert sel.sum()==630 and len(sel)==755
dx=xp.to_numpy()-xp.loc['center.bmp'].to_numpy();dy=yp.to_numpy()-yp.loc['center.bmp'].to_numpy()
def p12(pts,d):
    u=(pts[:,0]-1906)/486;v=(pts[:,1]-1070)/122;B=np.c_[u*0+1,u,v,u*u,u*v,v*v]
    return np.linalg.lstsq(B,d.transpose(1,0,2).reshape(len(pts),-1),rcond=None)[0].reshape(6,len(d),2).transpose(1,0,2).reshape(len(d),12)
points=ref[['x','y']].to_numpy()[sel];ds=np.stack([dx[:,sel],dy[:,sel]],axis=-1)
features={'corner630':p12(points,ds)}
old=np.load(BASE/'center_fixed_20261015/center_once_tracks.npz');on=list(old['names'].astype(str));features['old205']=p12(old['points'],np.stack([old['D'][on.index(n)] for n in names]).astype(float))
folds=json.loads((BASE/'gls_comparison_20261014/splits.json').read_text(encoding='utf-8'));ids=np.flatnonzero(m.axis!='C');preds=[];models={};case={}
for feature,X in features.items():
    for mode in ['center','inferred_session']:
        key=feature+'_'+mode
        for split in folds:
            tr=np.array([ix[n] for n in split['train']]);te=np.array([ix[n] for n in split['test']]);model=Z.fit(m,X,tr,m.value.to_numpy(),mode,True)
            p=Z.predict(m,X,te,model,mode).assign(method=key,protocol=split['protocol'],fold=split['fold']);preds.append(p)
            if split['protocol']=='full280':
                branches,W=model;bundle=dict(W=W.tolist(),branches={k:[v[0],*[q.tolist() for q in v[1:]]] for k,v in branches.items()},zero_feature=[0]*12,feature=feature,mode=mode)
                (OUT/(key+'_model.json')).write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
                if feature=='corner630' and mode=='inferred_session':
                    sys.path.insert(0,str(BASE));from infer import solve_calibrated,clean
                    case=clean(solve_calibrated(X[ix['Y+0.05.bmp']],bundle))
        print(key,flush=True)
P=pd.concat(preds,ignore_index=True);truth=np.c_[np.where(P.axis=='Y',P.value,0),np.where(P.axis=='T',P.value,0)];pr=np.c_[np.where(P.pred_axis=='Y',P.pred,0),np.where(P.pred_axis=='T',P.pred,0)]
P['eY']=abs(pr[:,0]-truth[:,0]);P['eT']=abs(pr[:,1]-truth[:,1]);P['correct']=(P.axis==P.pred_axis)&(np.sign(P.value)==np.sign(P.pred));P['pass']=P.correct&(P.eY<=.09)&(P.eT<=.09)
P.to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig');P[~P['pass']].to_csv(OUT/'failures.csv',index=False,encoding='utf-8-sig')
metrics=[]
for (method,protocol),g in P.groupby(['method','protocol']):metrics.append(dict(method=method,protocol=protocol,**Z.score(g)))
pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
summary=[]
for method,g in P.groupby('method'):
    for protocol,gg in [('full280',g[g.protocol=='full280']),('repeated20',g[g.protocol.str.startswith('repeated5_')]),('interleaved',g[g.protocol=='interleaved'])]:
        summary.append(dict(method=method,protocol=protocol,Y_correct=int(gg[gg.axis=='Y'].correct.sum()),Y_n=int((gg.axis=='Y').sum()),T_correct=int(gg[gg.axis=='T'].correct.sum()),T_n=int((gg.axis=='T').sum()),failures=int((~gg['pass']).sum()),Y_max=float(gg.eY.max()),T_max=float(gg.eT.max())))
S=pd.DataFrame(summary);S.to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
V=C.pivot(index='file',columns='id',values='valid').loc[names].to_numpy()[:,sel]
audit=dict(source_sha256=before,photos=len(names),ids_total=755,excluded_diagonal=125,used=630,id_rule='center black_neighbors !=2; same IDs in all281 frames',uncertain_rows=int((~V).sum()),valid_rate=float(V.mean()),exclude_by_validation=False,measurement='Decoded-grid blurred-edge model, not Shi-Tomasi/LK/GN17',model='same P12 and GLS fit/predict; all119 saved splits; training-only offsets and covariance',new_QR_payload_generalization_tested=False)
assert before==hashlib.sha256(source.read_bytes()).hexdigest()
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8');(OUT/'case630.json').write_text(json.dumps(case,indent=2,ensure_ascii=False),encoding='utf-8')
print(S.to_string(index=False),flush=True)

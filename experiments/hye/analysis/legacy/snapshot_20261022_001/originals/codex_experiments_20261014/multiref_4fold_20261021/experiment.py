from pathlib import Path
import sys,json,itertools
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;B=H.parent
sys.path.insert(0,str(B));import zero_experiment as Z
def fit(X,m,tr,mode,quad,shrink):
    branches,_=Z.fit(m,X,tr,m.value.to_numpy(),mode,quad)
    R=[]
    for i in tr:
        k=m.session.iloc[i] if mode=='inferred_session' else m.axis.iloc[i]
        a,b,s,c=branches[k];v=m.value.iloc[i];R.append(X[i]-b-s*v-c*v*v)
    R=np.array(R);S=R.T@R/max(len(R)-16,1);std=np.sqrt(np.maximum(np.diag(S),1e-14));C=S/std[:,None]/std[None,:]
    C=(1-shrink)*C+shrink*np.eye(len(C));W=np.linalg.cholesky(np.linalg.inv(C)).T/std[None,:]
    return branches,W
def enrich(p):
    p=p.copy();p['correct']=(p.axis==p.pred_axis)&(np.sign(p.value)==np.sign(p.pred))
    for a in ['Y','T']:p['e'+a]=abs(np.where(p.axis==a,p.value,0)-np.where(p.pred_axis==a,p.pred,0))
    p['pass']=p.correct&(p.eY<=.09)&(p.eT<=.09);return p
def stats(p):
    return dict(n=len(p),Y_correct=int(p[p.axis=='Y'].correct.sum()),Y_n=int((p.axis=='Y').sum()),T_correct=int(p[p.axis=='T'].correct.sum()),T_n=int((p.axis=='T').sum()),failures=int((~p['pass']).sum()),Y_max=float(p.eY.max()),T_max=float(p.eT.max()))
def main():
    global H
    source_dir=H
    if '--extended' in sys.argv:
        H=H/'extended';H.mkdir(exist_ok=True)
    if '--local' in sys.argv:
        H=H/'local_fields';H.mkdir(exist_ok=True)
    if '--subset' in sys.argv:
        H=H/'center_subsets';H.mkdir(exist_ok=True)
    if '--windows' in sys.argv:
        H=B/'patch_windows_20261021/evaluation';H.mkdir(exist_ok=True)
    if '--robust' in sys.argv:
        H=B/'robust_gn_20261021/evaluation';H.mkdir(exist_ok=True)
    if '--spatial' in sys.argv:
        H=H/'robust_spatial';H.mkdir(exist_ok=True)
    if '--screws' in sys.argv:
        H=H/'screw_fallback';H.mkdir(exist_ok=True)
    if '--forward' in sys.argv:
        H=B/'forward_gn_20261021/evaluation';H.mkdir(exist_ok=True)
    if '--regions' in sys.argv:
        H=H/'structural_regions';H.mkdir(exist_ok=True)
    cache=np.load(B/'center_patch_v2_20261021/tracks.npz');m=Z.manifest(cache['names']);ids=np.flatnonzero(m.axis!='C')
    features={k:np.load(B/'center_patch_v2_20261021'/f'{k}_features.npy') for k in ['v1_edge630','v2_patch630','v2_patch630_quality']}
    if '--extended' in sys.argv:
        for key in ['ecc_affine','ecc_homography','phase']:features[key]=np.load(source_dir/(key+'_features.npy'))
    if '--windows' in sys.argv:
        features={p.stem.replace('_features',''):np.load(p) for p in sorted((B/'patch_windows_20261021').glob('*_features.npy'))}
        assert len(features)==6
    if '--robust' in sys.argv:
        features={p.stem.replace('_features',''):np.load(p) for p in sorted((B/'robust_gn_20261021').glob('*_features.npy'))}
        assert len(features)==4
    if '--forward' in sys.argv:
        features={p.stem.replace('_features',''):np.load(p) for p in sorted((B/'forward_gn_20261021').glob('*_features.npy'))}
        assert len(features)==4
    if '--regions' in sys.argv:
        C=pd.read_csv(B.parent/'analysis_grid_corners_20261019/out/corners_280.csv');ref=C[C.file=='center.bmp'].set_index('id').loc[cache['ids']];gx=ref.grid_x.to_numpy();gy=ref.grid_y.to_numpy()
        finder=((gx<=7)&(gy<=7))|((gx>=14)&(gy<=7))|((gx<=7)&(gy>=14));interior=(gx>0)&(gx<21)&(gy>0)&(gy<21)
        features={};pts=cache['points'];D=cache['D'].astype(float);D-=D[0];audit=[]
        for key,sel in [('finder',finder),('data',~finder),('interior',interior)]:
            p=pts[sel];u=(p[:,0]-1906)/486;v=(p[:,1]-1070)/122;design=np.c_[u*0+1,u,v,u*u,u*v,v*v];d=D[:,sel]
            features[key]=np.linalg.lstsq(design,d.transpose(1,0,2).reshape(len(p),-1),rcond=None)[0].reshape(6,len(d),2).transpose(1,0,2).reshape(len(d),12)
            np.save(H/(key+'_features.npy'),features[key]);audit.append(dict(region=key,count=int(sel.sum()),ids=cache['ids'][sel].tolist()))
        (H/'selection_audit.json').write_text(json.dumps(dict(rule='Actual QR21-module grid corners; finder x<=7 y<=7 OR x>=14 y<=7 OR x<=7 y>=14. Data complement; interior excludes grid0/21. Selection uses center IDs only.',regions=audit),indent=2))
    if '--screws' in sys.argv:
        sdata=np.load(B/'measurement/gn17.npz');sn=list(sdata['names'].astype(str));D=sdata['D'].astype(float);D-=D[sn.index('center.bmp')];D=D[[sn.index(n) for n in m.file]]
        qr={'gn17':np.load(B/'center_patch_v2_20261021/v2_patch630_features.npy'),'gn31':np.load(B/'patch_windows_20261021/gn31_sigma0.8_features.npy')};features={}
        for agg in ['mean','median']:
            points=np.concatenate([getattr(np,agg)(D[:,sdata['regions']==r],axis=1) for r in [3,4]],axis=1)
            for key,X in qr.items():features[key+'_screws_'+agg]=np.c_[X,points]
        (H/'measurement_audit.json').write_text(json.dumps(dict(source='measurement/gn17.npz',screw_point_counts=[int((sdata['regions']==r).sum()) for r in [3,4]],valid_fraction=float(sdata['valid'][:,sdata['regions']>=3].mean()),scope='EXPLICIT screw fallback, not QR-only. Same630 QR IDs plus4 screw displacement summaries. Existing measurement cache reused, no fresh detection in this experiment.'),indent=2))
        for key,X in features.items():np.save(H/(key+'_features.npy'),X)
    if '--spatial' in sys.argv:
        features={};pts=cache['points'];u=(pts[:,0]-1906)/486;v=(pts[:,1]-1070)/122;design=np.c_[u*0+1,u,v,u*u,u*v,v*v]
        sources={'gn17':cache['D'].astype(float),'gn31':np.load(B/'patch_windows_20261021/gn31_sigma0.8_tracks.npz')['D'].astype(float)}
        for src,D in sources.items():
            D=D-D[0]
            for delta in [.01,.03,.1,.3]:
                out=[]
                for d in D:
                    coef=np.linalg.lstsq(design,d,rcond=None)[0]
                    for _ in range(5):
                        err=np.linalg.norm(d-design@coef,axis=1);w=np.minimum(1,delta/np.maximum(err,1e-12));sw=np.sqrt(w)
                        coef=np.linalg.lstsq(design*sw[:,None],d*sw[:,None],rcond=None)[0]
                    out.append(coef.ravel())
                key=f'{src}_spatialHuber{delta}';features[key]=np.array(out);np.save(H/(key+'_features.npy'),features[key])
    if '--local' in sys.argv:
        features={}
        D=cache['D'].astype(float);D-=D[0];pts=cache['points'];regs=np.argmin(abs(pts[:,0,None]-[1644,1901,2159]),axis=1)
        for degree in [1,2]:
            for weighted in [False,True]:
                chunks=[]
                for region in range(3):
                    sel=regs==region;x=(pts[sel,0]-[1644,1901,2159][region])/100;y=(pts[sel,1]-1070)/100
                    design=np.c_[x*0+1,x,y] if degree==1 else np.c_[x*0+1,x,y,x*x,x*y,y*y]
                    out=[]
                    for i,d in enumerate(D[:,sel]):
                        q=cache['quality'][i,sel];w=1/(1+(q[:,0]/6)**2+(q[:,4]/.15)**2) if weighted else np.ones(len(d))
                        out.append(np.linalg.lstsq(design*np.sqrt(w)[:,None],d*np.sqrt(w)[:,None],rcond=None)[0].ravel())
                    chunks.append(np.array(out))
                key=f'local_degree{degree}_weighted{weighted}';features[key]=np.concatenate(chunks,axis=1);np.save(H/(key+'_features.npy'),features[key])
    if '--subset' in sys.argv:
        import cv2
        features={};D=cache['D'].astype(float);D-=D[0];pts=cache['points']
        img=cv2.imdecode(np.fromfile(Z.ROOT/'center.bmp',np.uint8),0)
        eig=cv2.cornerMinEigenVal(cv2.GaussianBlur(img,(0,0),.8),5)
        scores=eig[np.rint(pts[:,1]).astype(int),np.rint(pts[:,0]).astype(int)]
        for spacing in [8,12,16,20]:
            chosen=[]
            for i in np.argsort(-scores,kind='stable'):
                if not chosen or np.min(np.linalg.norm(pts[chosen]-pts[i],axis=1))>=spacing:chosen.append(int(i))
            p=pts[chosen];u=(p[:,0]-1906)/486;v=(p[:,1]-1070)/122;design=np.c_[u*0+1,u,v,u*u,u*v,v*v]
            d=D[:,chosen];key=f'center_spacing{spacing}_n{len(chosen)}'
            features[key]=np.linalg.lstsq(design,d.transpose(1,0,2).reshape(len(chosen),-1),rcond=None)[0].reshape(6,len(d),2).transpose(1,0,2).reshape(len(d),12)
            np.save(H/(key+'_features.npy'),features[key]);(H/(key+'_ids.json')).write_text(json.dumps(dict(indices=chosen,structural_ids=cache['ids'][chosen].tolist(),selection='center-only minimum eigenvalue descending, minimum pixel distance; no current-image or label selection'),indent=2))
    configs=list(itertools.product(features,['center','inferred_session'],[False,True],[.001,.05,.2,1.]))
    folds=list(StratifiedKFold(4,shuffle=True,random_state=20261021).split(ids,m.session.iloc[ids]))
    (H/'splits.json').write_text(json.dumps([dict(fold=f,train=m.file.iloc[ids[a]].tolist(),test=m.file.iloc[ids[b]].tolist()) for f,(a,b) in enumerate(folds)],indent=2))
    configmap={f'c{k:02d}':dict(feature=c[0],mode=c[1],quadratic=c[2],shrinkage=c[3]) for k,c in enumerate(configs)}
    (H/'configs.json').write_text(json.dumps(configmap,indent=2));allrows=[];nested=[];selection=[]
    for f,(itr,ite) in enumerate(folds):
        tr=ids[itr];te=ids[ite];inner=list(StratifiedKFold(3,shuffle=True,random_state=100+f).split(tr,m.session.iloc[tr]));rank=[]
        for k,(feature,mode,quad,shrink) in enumerate(configs):
            key=f'c{k:02d}';X=features[feature];innerp=[]
            for ia,ib in inner:
                model=fit(X,m,tr[ia],mode,quad,shrink);innerp.append(Z.predict(m,X,tr[ib],model,mode))
            ip=enrich(pd.concat(innerp));ss=stats(ip);rank.append((ss['failures'],max(ss['Y_max'],ss['T_max']),k))
            model=fit(X,m,tr,mode,quad,shrink);p=enrich(Z.predict(m,X,te,model,mode)).assign(config=key,protocol='oof4',fold=f);allrows.append(p)
        chosen=min(rank)[2];nested.append(allrows[-len(configs)+chosen].assign(config='nested_selected'))
        selection.append(dict(fold=f,selected=f'c{chosen:02d}',inner_rank=sorted(rank)))
        print('outer fold',f,'selected',chosen,flush=True)
    for k,(feature,mode,quad,shrink) in enumerate(configs):
        model=fit(features[feature],m,ids,mode,quad,shrink);allrows.append(enrich(Z.predict(m,features[feature],ids,model,mode)).assign(config=f'c{k:02d}',protocol='resub280',fold=-1))
        branches,W=model;(H/f'c{k:02d}_model.json').write_text(json.dumps(dict(W=W.tolist(),branches={b:[v[0],*[x.tolist() for x in v[1:]]] for b,v in branches.items()})))
    p=pd.concat(allrows+nested,ignore_index=True);p.to_csv(H/'predictions.csv',index=False);p[~p['pass']].to_csv(H/'failures.csv',index=False)
    out=[]
    for (key,protocol),g in p.groupby(['config','protocol']):out.append(dict(config=key,protocol=protocol,**configmap.get(key,{}),**stats(g)))
    s=pd.DataFrame(out);s.to_csv(H/'summary.csv',index=False);(H/'nested_selection.json').write_text(json.dumps(selection,indent=2))
    foldstats=pd.DataFrame([dict(config=k,fold=f,**stats(g)) for (k,f),g in p[p.protocol=='oof4'].groupby(['config','fold'])]);foldstats.to_csv(H/'fold_results.csv',index=False)
    print(s[s.protocol=='oof4'].sort_values(['failures','Y_max']).head(12).to_string(index=False),flush=True)
    print('NESTED',s[s.config=='nested_selected'].to_string(index=False),flush=True)
    (H/'report.html').write_text('<meta charset="utf-8"><style>body{font:16px sans-serif;margin:40px}td,th{padding:8px;border:1px solid #ddd}</style><h1>4-fold / 복수 원점 후보 비교</h1><p>측정 '+str(len(features))+'종 × 단일/복수 원점 × 선형/이차 × 공분산 축소 4종 = '+str(len(configs))+'설정. 시험 70장씩 총280장. nested_selected는 각 학습210장 내부3-fold로 설정 선택. 모든 데이터는 기존 시행이며 새 촬영 일반화 성적이 아님.</p>'+s.sort_values(['protocol','failures','Y_max']).to_html(index=False)+'<h2>Fold별 결과</h2>'+foldstats.to_html(index=False),encoding='utf-8')
if __name__=='__main__':main()

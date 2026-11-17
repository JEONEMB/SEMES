"""Information-preserving joint linear track, all630 coordinates, no axis gate."""
import json,joblib
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from engine import H,B,load_observations
from experiment import Study,metrics,rank_score

def field_features(points,d,q,kind):
    if kind=='raw':return d.ravel()
    if kind=='raw_quality':
        regs=np.argmin(abs(points[:,0,None]-[1644,1901,2159]),axis=1)
        summaries=np.concatenate([np.quantile(q[regs==r],[.1,.5,.9],axis=0).ravel() for r in range(3)])
        return np.r_[d.ravel(),summaries]
    bins=int(kind[4:]);regs=np.argmin(abs(points[:,0,None]-[1644,1901,2159]),axis=1);chunks=[]
    for r in range(3):
        sel=regs==r;p=points[sel];dd=d[sel]
        ix=np.minimum(((p-p.min(0))/(p.max(0)-p.min(0)+1e-9)*bins).astype(int),bins-1)
        for a in range(bins):
            for b in range(bins):
                mask=(ix[:,0]==a)&(ix[:,1]==b)
                if mask.any():chunks.extend(dd[mask].mean(0))
    return np.array(chunks)

class JointLinear:
    def __init__(self,c):self.c=c
    def fit(self,X,y):
        c=self.c;f=c['family'];self.origin=None
        if f=='gls_full':
            co=np.linalg.lstsq(np.c_[np.ones(len(y)),y],X,rcond=None)[0];r=X-np.c_[np.ones(len(y)),y]@co
            std=np.maximum(np.sqrt(np.mean(r*r,axis=0)),1e-7);rr=r/std
            shrink=c['alpha'];a=shrink;b=(1-shrink)/len(y)
            s=(co[1:]/std).T
            # Woodbury avoids a dense1260x1260 inversion. Metric aI+bR'R.
            Q=(s-b*rr.T@np.linalg.solve(a*np.eye(len(y))+b*rr@rr.T,rr@s))/a
            self.beta=Q@np.linalg.inv((co[1:]/std)@Q)/std[:,None];self.origin=co[0];self.model=None
        else:
            if f=='ridge':self.model=make_pipeline(StandardScaler(),Ridge(alpha=c['alpha']))
            elif f=='pcr':self.model=make_pipeline(StandardScaler(),PCA(n_components=min(c['components'],X.shape[1]),svd_solver='full'),Ridge(alpha=.01))
            elif f=='pls':self.model=make_pipeline(StandardScaler(),PLSRegression(n_components=min(c['components'],X.shape[1]),scale=False,max_iter=600))
            self.model.fit(X,y)
        return self
    def predict(self,X):return (X-self.origin)@self.beta if self.model is None else self.model.predict(X)

def configs():
    models=[dict(family='ridge',alpha=a) for a in [.001,.01,.1,1.,10.,100.]]
    models+=[dict(family='gls_full',alpha=a) for a in [.001,.01,.1,.5,1.]]
    models+=[dict(family=f,components=k) for f in ['pcr','pls'] for k in [4,8,16,32]]
    return [dict(feature=feat,**m) for feat in ['grid2','grid4','grid8','raw','raw_quality'] for m in models]

def main():
    s=Study();o=s.obs;cache={}
    for kind in ['grid2','grid4','grid8','raw','raw_quality']:
        cache[kind]=np.stack([field_features(o['points'],d,q,kind) for d,q in zip(o['D'],o['q'])]);print('full field',kind,cache[kind].shape,flush=True)
    registry=configs();(H/'fullfield_registry.json').write_text(json.dumps(registry,indent=2))
    folds=json.loads((H/'splits.json').read_text());search=[];selections=[];preds=[];synths=[];models=[]
    for fold in range(5):
        if fold==4:tr=s.ids;te=None
        else:
            tr=np.array([s.lookup[n] for n in folds[fold]['train']]);te=np.array([s.lookup[n] for n in folds[fold]['test']])
        inner=list(StratifiedKFold(3,shuffle=True,random_state=1200+fold).split(tr,s.m.session.iloc[tr]));ranks=[]
        for k,c in enumerate(registry):
            X=cache[c['feature']];ip=[];it=[]
            for a,b in inner:
                m=JointLinear(c).fit(X[tr[a]],s.y[tr[a]]);ip.append(m.predict(X[tr[b]]));it.append(s.y[tr[b]])
            score=rank_score(np.vstack(ip),np.vstack(it));search.append(dict(fold=fold,candidate=k,config=c,inner_score=score));ranks.append((*score,k))
        chosen=min(ranks)[-1];c=registry[chosen];X=cache[c['feature']];model=JointLinear(c).fit(X[tr],s.y[tr]);bundle=dict(config=c,model=model,ids=o['ids'],points=o['points'])
        selections.append(dict(fold=fold,config=c,inner_score=list(min(ranks)[:3])))
        if te is None:joblib.dump(bundle,H/'fullfield_model.joblib');continue
        joblib.dump(bundle,H/f'fullfield_fold{fold}.joblib');p=model.predict(X[te]);truth=s.y[te]
        preds.append(pd.DataFrame(dict(file=s.names[te],axis=s.m.axis.iloc[te].to_numpy(),fold=fold,true_Y=truth[:,0],true_T=truth[:,1],pred_Y=p[:,0],pred_T=p[:,1],both_pass09=np.all(abs(p-truth)<=.09,axis=1))))
        yi=te[s.y[te,0]!=0];ti=te[s.y[te,1]!=0];ii=np.repeat(yi,len(ti));jj=np.tile(ti,len(yi))
        origin=np.linalg.lstsq(np.c_[np.ones(len(tr)),s.y[tr],s.y[tr]**2],X[tr],rcond=None)[0][0]
        pp=model.predict(X[ii]+X[jj]-origin);tt=s.y[ii]+s.y[jj]
        synths.append(pd.DataFrame(dict(fold=fold,Y_source=s.names[ii],T_source=s.names[jj],true_Y=tt[:,0],true_T=tt[:,1],pred_Y=pp[:,0],pred_T=pp[:,1])))
        print('fullfield selected',fold,c,'inner',min(ranks)[:3], 'outer',metrics(p,truth),flush=True)
        (H/'fullfield_progress.json').write_text(json.dumps(selections,indent=2))
    p=pd.concat(preds,ignore_index=True);p.to_csv(H/'fullfield_oof.csv',index=False);sy=pd.concat(synths,ignore_index=True);sy.to_csv(H/'fullfield_synthetic.csv',index=False)
    report=dict(selection=selections,real_oof=metrics(p[['pred_Y','pred_T']].to_numpy(),p[['true_Y','true_T']].to_numpy()),synthetic_mixed=metrics(sy[['pred_Y','pred_T']].to_numpy(),sy[['true_Y','true_T']].to_numpy()),limits='Same historical folds, no real mixed images. raw_quality includes QR photometric statistics and is separately identified.')
    (H/'fullfield_summary.json').write_text(json.dumps(report,indent=2));(H/'fullfield_search.json').write_text(json.dumps(search,indent=2))
    summary=[]
    for key,g in p.groupby('axis'):summary.append(dict(subset=key,**metrics(g[['pred_Y','pred_T']].to_numpy(),g[['true_Y','true_T']].to_numpy())))
    path=H/'report.html';text=path.read_text(encoding='utf-8');text+='<h2>630점 전체 정보 추가 실험</h2><p>공간 요약으로 약한 구분 단서를 잃는지 검사했다. 2×2/4×4/8×8 QR별 집계, 전체1260 좌표, 좌표+QR 품질 통계를 비교하고 각 표현에 Ridge·PCR·PLS·전체 공분산 GLS를 적용했다. 95개 설정은 각 학습 내부에서 선택했다. 전체 점을 유지하며 가산 선형 출력이다. 별도 fullfield_model.joblib에 저장.</p><p>OOF: '+str(report['real_oof'])+'</p><p>합성 혼합: '+str(report['synthetic_mixed'])+'</p>'+pd.DataFrame(summary).to_html(index=False)
    path.write_text(text,encoding='utf-8');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    # Use importable class identity in serialized bundles.
    from fullfield import JointLinear
    with threadpool_limits(limits=1):main()

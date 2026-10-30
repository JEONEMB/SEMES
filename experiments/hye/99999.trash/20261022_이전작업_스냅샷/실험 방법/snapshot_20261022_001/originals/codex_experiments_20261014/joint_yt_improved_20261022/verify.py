"""Checks for whitening, real feature parity, fold isolation, and image inference."""
import sys,json
import numpy as np
import pandas as pd
import joblib
from engine import *

def main():
    checks={};rng=np.random.default_rng(4)
    # Numerical identity: row-vector whitening must equal inverse covariance metric.
    y=rng.normal(size=(90,2));s=rng.normal(size=(3,6));r=rng.normal(size=(90,6))@np.diag([.01,.03,.2,1,3,10]);X=np.c_[np.ones(90),y]@s+r
    f=Forward().fit(X,y);R=X-f.design(y)@f.co;std=np.sqrt(np.mean(R*R,axis=0));C=.999*(R/std).T@(R/std)/len(R)+.001*np.eye(6)
    S=std[:,None]*C*std[None,:]
    assert np.allclose(f.W@f.W.T,np.linalg.inv(S),rtol=1e-8,atol=1e-8)
    inv=np.linalg.inv(S);jac=f.co[1:];expected=(X-f.co[0])@inv@jac.T@np.linalg.inv(jac@inv@jac.T)
    assert np.allclose(f.predict(X),expected,atol=1e-8)
    checks['whitening_matches_GLS']=True
    obs=load_observations();baseline=np.load(B/'center_patch_v2_20261021/v1_edge630_features.npy')
    actual=np.stack([extract(obs['points'],d,q,'uniform','global2') for d,q in zip(obs['D'],obs['q'])])
    delta=float(abs(actual-baseline).max());assert delta<1e-8;checks['original_feature_max_delta']=delta
    folds=json.loads((H/'splits.json').read_text());tests=[]
    for s in folds:
        assert len(s['train'])==210 and len(s['test'])==70 and not set(s['train'])&set(s['test']);tests+=s['test']
    assert len(tests)==len(set(tests))==280;checks['fold_partition_isolation']=True
    p=pd.read_csv(H/'predictions.csv');g=p[p.method=='improved_nested'];assert len(g)==g.file.nunique()==280
    true=g[['true_Y','true_T']].to_numpy();pred=g[['pred_Y','pred_T']].to_numpy()
    assert np.array_equal(np.all(abs(true-pred)<=.09,axis=1),g.both_pass09.to_numpy());checks['independent_error_recalculation']=True
    # Each recorded inner selection must minimize the stored inner-only score.
    search=json.loads((H/'search.json').read_text());selections=json.loads((H/'selection.json').read_text())
    for s in selections:
        same=[r for r in search if r['fold']==s['fold'] and r['stage']==s['stage']]
        best=min(same,key=lambda r:(*r['inner_score'],r['candidate']))
        assert best['candidate']==s['chosen']
    checks['selection_matches_inner_scores']=True
    if '--images' in sys.argv:
        from predict import predict
        sys.path.insert(0,str(B));import zero_experiment as Z
        bundle=joblib.load(H/'model.joblib');c=bundle['config'];image_checks=[]
        for name in ['Y+0.05.bmp','T-0.05.bmp']:
            raw=(Z.ROOT/('Y' if name[0]=='Y' else 'Theta')/name).read_bytes()
            result=predict(raw,(Z.ROOT/'center.bmp').read_bytes());idx=list(obs['names']).index(name)
            feature=extract(obs['points'],obs['D'][idx],obs['q'][idx],c['measurement'],c['feature']);expected=bundle['model'].predict(feature[None])[0]
            delta=float(abs(feature-result['feature']).max());error=float(abs(expected-[result['Y'],result['T']]).max())
            assert delta<1e-8 and error<1e-7,(delta,error)
            image_checks.append(dict(source=name,feature_max_delta=delta,prediction_max_delta=error,result=result))
        checks['image_byte_inference']=image_checks
    (H/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()

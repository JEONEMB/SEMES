"""Select full-field linear models without photometric quality predictors."""
import sys,json,joblib
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from engine import *
from fullfield import JointLinear,field_features
from experiment import Study,metrics

def main():
    # Repair import identity of bundles produced by the first direct-script run.
    import __main__
    __main__.JointLinear=JointLinear
    for path in H.glob('fullfield*joblib'):
        bundle=joblib.load(path);joblib.dump(bundle,path)
    s=Study();search=json.loads((H/'fullfield_search.json').read_text());folds=json.loads((H/'splits.json').read_text());cache={};selections=[];out=[];synth=[]
    for fold in range(5):
        possible=[r for r in search if r['fold']==fold and r['config']['feature']!='raw_quality']
        chosen=min(possible,key=lambda r:(*r['inner_score'],r['candidate']));c=chosen['config'];selections.append(chosen)
        if c['feature'] not in cache:cache[c['feature']]=np.stack([field_features(s.obs['points'],d,q,c['feature']) for d,q in zip(s.obs['D'],s.obs['q'])])
        X=cache[c['feature']]
        if fold==4:tr=s.ids;te=None
        else:tr=np.array([s.lookup[n] for n in folds[fold]['train']]);te=np.array([s.lookup[n] for n in folds[fold]['test']])
        model=JointLinear(c).fit(X[tr],s.y[tr]);joblib.dump(dict(config=c,model=model,ids=s.obs['ids'],points=s.obs['points']),H/('coordinate_model.joblib' if te is None else f'coordinate_fold{fold}.joblib'))
        if te is None:continue
        p=model.predict(X[te]);y=s.y[te];out.append(pd.DataFrame(dict(file=s.names[te],axis=s.m.axis.iloc[te].to_numpy(),fold=fold,true_Y=y[:,0],true_T=y[:,1],pred_Y=p[:,0],pred_T=p[:,1],both_pass09=np.all(abs(p-y)<=.09,axis=1))))
        yi=te[s.y[te,0]!=0];ti=te[s.y[te,1]!=0];ii=np.repeat(yi,len(ti));jj=np.tile(ti,len(yi))
        origin=np.linalg.lstsq(np.c_[np.ones(len(tr)),s.y[tr],s.y[tr]**2],X[tr],rcond=None)[0][0]
        pp=model.predict(X[ii]+X[jj]-origin);tt=s.y[ii]+s.y[jj]
        synth.append(pd.DataFrame(dict(fold=fold,true_Y=tt[:,0],true_T=tt[:,1],pred_Y=pp[:,0],pred_T=pp[:,1])))
    p=pd.concat(out,ignore_index=True);sy=pd.concat(synth,ignore_index=True);p.to_csv(H/'coordinate_oof.csv',index=False);sy.to_csv(H/'coordinate_synthetic.csv',index=False)
    report=dict(real_oof=metrics(p[['pred_Y','pred_T']].to_numpy(),p[['true_Y','true_T']].to_numpy()),synthetic_mixed=metrics(sy[['pred_Y','pred_T']].to_numpy(),sy[['true_Y','true_T']].to_numpy()),selection=selections)
    (H/'coordinate_summary.json').write_text(json.dumps(report,indent=2))
    # Check Woodbury GLS against a direct covariance inverse on a small test system.
    rng=np.random.default_rng(7);y=rng.normal(size=(60,2));X=rng.normal(size=(60,16))+y@rng.normal(size=(2,16))
    model=JointLinear(dict(family='gls_full',alpha=.1)).fit(X,y);A=np.c_[np.ones(len(y)),y];co=np.linalg.lstsq(A,X,rcond=None)[0];r=X-A@co
    std=np.maximum(np.sqrt(np.mean(r*r,axis=0)),1e-7);C=.9*r.T@r/len(r)+.1*np.diag(std*std);Q=np.linalg.solve(C,co[1:].T);beta=Q@np.linalg.inv(co[1:]@Q)
    assert np.allclose(model.predict(X),(X-co[0])@beta,atol=1e-10)
    report['woodbury_dense_equivalence']=True
    (H/'coordinate_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()

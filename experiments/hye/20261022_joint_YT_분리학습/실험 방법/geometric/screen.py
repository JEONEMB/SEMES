from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;PREV=H.parents[1]/'joint_yt_generalization_20261022'
sys.path.insert(0,str(PREV))
from data import load,metric
from geometric_models import RepeatGLS,geometry_features

def main():
    m,a=load();D=a['D'];y=m[['Y','T']].to_numpy();new=m.dataset.to_numpy()==20261020
    bank={kind:geometry_features(D,a['points'],a['regions'],kind) for kind in ['raw','raw_shift','shape','shape_shift','strain','moments']}
    np.savez_compressed(H/'features.npz',**bank)
    configs=[dict(feature=f,fit_mode=mode,repeat_mix=mix,shrink=s) for f in bank for mode in ['all','new_condition_means','axis_pairs'] for mix in [0.,.5,1.] for s in [.01,.1,.5]]
    protocol=dict(source=str(PREV/'measurements/features.npz'),folds=m.fold.tolist(),configs=configs,scope='development grouped4fold, all new centers samefold',labels='training-only forward response / repetition covariance; no filename at prediction',references=[{'title':'Robust image-based visual servoing using invariant visual information','url':'https://www.sciencedirect.com/science/article/abs/pii/S092188901300119X','use':'motivates translation-invariant centered geometric features; no exact pose reconstruction assumed'}])
    (H/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];predictions=[];audits=[]
    for k,c in enumerate(configs):
        p=np.zeros_like(y);X=bank[c['feature']];audit=[]
        for f in range(4):
            tr=m.fold.to_numpy()!=f;te=~tr
            assert not set(m.cv_group[tr])&set(m.cv_group[te])
            model=RepeatGLS(c['shrink'],c['repeat_mix'],c['fit_mode']).fit(X[tr],y[tr],m.cv_group.to_numpy()[tr],new[tr]);p[te]=model.predict(X[te]);audit.append(dict(fold=f,repeat_dof=model.repeat_dof,response=model.response.tolist()))
        for subset,mask in [('all',np.ones(len(m),bool)),('new',new),('old',~new)]:rows.append(dict(candidate=k,**c,subset=subset,**metric(y[mask],p[mask])))
        z=m[['index','dataset','file','Y','T','scope','condition','fold']].copy();z['candidate']=k;z['predY']=p[:,0];z['predT']=p[:,1];predictions.append(z)
        if k%27==0:print(k,c,metric(y[new],p[new]),flush=True)
        audits.append(dict(candidate=k,folds=audit))
    pd.DataFrame(rows).to_csv(H/'summary.csv',index=False);pd.concat(predictions).to_csv(H/'predictions.csv',index=False)
    (H/'response_audit.json').write_text(json.dumps(audits,indent=2))
    s=pd.DataFrame(rows);print(s[s.subset=='new'].sort_values(['passed','Y_max'],ascending=[False,True]).head(20).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()

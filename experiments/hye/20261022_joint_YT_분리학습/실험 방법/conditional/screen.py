from pathlib import Path
import json,sys,time
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;PREV=H.parents[1]/'joint_yt_generalization_20261022'
sys.path.insert(0,str(PREV))
from data import load,features,metric
from conditional_model import ConditionalDecoder,NuisanceProjected

def main():
    m,a=load();bank=features(a);y=m[['Y','T']].to_numpy();D=a['D'];new=m.dataset==20261020
    configs=[dict(feature=f,base='raw_shift',base_shrink=.1,mode=mode,family='gls',shrink=s) for f in ['P12','raw1260'] for mode in ['append_t','remove_linear','remove_quadratic','remove_cross','conditional_residual'] for s in [.01,.1,.5]]
    configs += [dict(feature=f,mode='project',degree=d,shrink=s) for f in ['P12','raw1260','raw_shift'] for d in [1,2] for s in [.01,.1,.5]]
    (H/'protocol.json').write_text(json.dumps(dict(configs=configs,measurement=str(PREV/'measurements/features.npz'),grouped=True,new_center3_same_fold=True,training_oof_nuisance=True,scope='development candidate comparison; not independent future-date test'),indent=2))
    rows=[];preds=[];audit=[]
    for k,c in enumerate(configs):
        started=time.monotonic();X=bank[c['feature']];p=np.full(y.shape,np.nan)
        for f in range(4):
            tr=np.flatnonzero(m.fold!=f);te=np.flatnonzero(m.fold==f)
            if c['mode']=='project':
                model=NuisanceProjected(c['shrink'],c['degree']).fit(X[tr],y[tr]);p[te]=model.predict(X[te])
            else:
                model=ConditionalDecoder(c).fit(X[tr],bank[c['base']][tr],y[tr],m.cv_group.to_numpy()[tr],np.where(m.dataset.to_numpy()[tr]==20261020,'new','old'))
                p[te]=model.predict(X[te],bank[c['base']][te])
                if k==0:audit.append(dict(outer=f,train=tr.tolist(),test=te.tolist(),nuisance_splits=[dict(train=tr[aa].tolist(),validation=tr[bb].tolist()) for aa,bb in model.crossfit_splits]))
        for scope,mask in [('new60',new),('old281',~new),('all341',np.ones(len(m),bool))]:rows.append(dict(candidate=k,**c,scope=scope,seconds=time.monotonic()-started,**metric(y[mask],p[mask])))
        pp=m[['index','dataset','file','Y','T','scope','condition','fold']].copy();pp['candidate']=k;pp['predY']=p[:,0];pp['predT']=p[:,1];preds.append(pp)
        if k%5==0:print(k,c,metric(y[new],p[new]),flush=True)
    pd.DataFrame(rows).to_csv(H/'summary.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(H/'predictions.csv',index=False);(H/'audit.json').write_text(json.dumps(audit,indent=2))
    df=pd.DataFrame(rows);print(df[df.scope=='new60'].sort_values(['passed','Y_max'],ascending=[False,True]).head(12).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()

"""Development screen, grouping all repetitions of each mixed command condition."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from forward_model import ForwardGLS
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
sys.path.insert(0,str(ROOT/'joint_center_630_20261022'))
from features import FeatureMaps

def metrics(y,p):
    e=abs(y-p);return dict(n=len(y),passed=int(np.all(e<=.09+1e-12,axis=1).sum()),Y_MAE=float(e[:,0].mean()),Y_max=float(e[:,0].max()),T_MAE=float(e[:,1].mean()),T_max=float(e[:,1].max()))

def main():
    data=ROOT/'joint_center_630_20261022/results';m=pd.read_csv(data/'manifest.csv');a=np.load(data/'measurements.npz');maps=FeatureMaps(a['points'],a['regions'])
    bank={f:maps.transform(a['D'],f) for f in ['P12','QR18','raw1260']}
    splits=json.loads((data/'expanded_features/splits.json').read_text());fold=np.full(len(m),-1)
    for f,s in enumerate(splits):fold[m.relative.isin(s['test'])]=f
    new=m.relative.str.startswith('20261020').to_numpy();y=m[['Y','T']].to_numpy()
    configs=[dict(feature=f,degree=d,shrink=s) for f in bank for d in [1,2,3] for s in [.001,.01,.1,.5,1.]]
    (H/'forward_protocol.json').write_text(json.dumps(dict(configs=configs,source=str(data),protocol='development grouped4fold; model selection after comparison is not independent'),indent=2))
    records=[];predictions=[]
    for regime in ['combined','new_only','new_weight5','old_transfer']:
        for k,c in enumerate(configs):
            pred=np.full(y.shape,np.nan);X=bank[c['feature']]
            for f in ([-1] if regime=='old_transfer' else range(4)):
                tr=(~new) if f==-1 else ((fold!=f)&(new if regime=='new_only' else True))
                te=new if f==-1 else ((fold==f)&new)
                weights=np.where(new[tr],5.,1.) if regime=='new_weight5' else None
                model=ForwardGLS(c['degree'],c['shrink']).fit(X[tr],y[tr],weights)
                pred[te]=model.predict(X[te])
            stats=metrics(y[new],pred[new]);records.append(dict(regime=regime,candidate=k,**c,**stats))
            pp=m.loc[new,['index','file','relative','Y','T','scope','condition']].copy();pp['regime']=regime;pp['candidate']=k;pp['fold']=fold[new];pp['predY']=pred[new,0];pp['predT']=pred[new,1];predictions.append(pp)
            if k%9==0:print(regime,k,stats,flush=True)
        pd.DataFrame(records).to_csv(H/'forward_screen_summary.csv',index=False);pd.concat(predictions).to_csv(H/'forward_screen_predictions.csv',index=False)
        print(pd.DataFrame(records).sort_values(['passed','Y_max'],ascending=[False,True]).head(5).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()

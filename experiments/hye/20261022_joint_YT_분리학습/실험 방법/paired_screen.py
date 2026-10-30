from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;PREV=H.parent/'joint_yt_generalization_20261022'
sys.path.insert(0,str(PREV))
from data import load,features,metric
from paired_model import PairedRidge

def main():
    m,a=load();bank=features(a);y=m[['Y','T']].to_numpy();new=m.dataset.to_numpy()==20261020;dates=m.dataset.to_numpy()
    configs=[dict(feature=f,alpha=al,pair_weight=pw,repeat_weight=rw) for f in ['P12','raw1260','raw_shift'] for al in [.0001,.01,1.] for pw,rw in [(0,0),(1,0),(10,0),(1,1),(1,10),(10,10)]]
    records=[];preds=[]
    for k,c in enumerate(configs):
        X=bank[c['feature']];p=np.full(y.shape,np.nan)
        for f in range(4):
            tr=m.fold.to_numpy()!=f;te=~tr
            model=PairedRidge(c).fit(X[tr],y[tr],dates[tr]);p[te]=model.predict(X[te])
        for subset,mask in [('new60',new),('old281',~new),('all341',np.ones(len(y),bool))]:records.append(dict(candidate=k,subset=subset,**c,**metric(y[mask],p[mask])))
        pp=m[['index','dataset','file','Y','T','scope','condition','fold']].copy();pp['candidate']=k;pp['predY']=p[:,0];pp['predT']=p[:,1];preds.append(pp)
        if k%9==0:print(k,records[-3],flush=True)
    pd.DataFrame(records).to_csv(H/'paired_summary.csv',index=False);pd.concat(preds).to_csv(H/'paired_predictions.csv',index=False)
    (H/'paired_protocol.json').write_text(json.dumps(dict(configs=configs,source=str(PREV),folds='same previous grouped4fold; newcenter3 together',training='sameT/sameY pair graph and within-command repeat graph computed only from training labels',inference='coordinate features only',scope='development screen'),indent=2))
    print(pd.DataFrame(records).query("subset=='new60'").sort_values(['passed','Y_max'],ascending=[False,True]).head(10).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()

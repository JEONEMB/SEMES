from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
from data import H,ROOT,load,features,metric,save_splits
sys.path.insert(0,str(ROOT/'codex_experiments_20261014/joint_yt_improved_20261022'))
from fullfield import JointLinear

def main():
    m,a=load();save_splits(m);bank=features(a);new=m.dataset==20261020;y=m[['Y','T']].to_numpy();rows=[];preds=[]
    configs=[dict(feature=f,C=c,epsilon=e) for f in ['P12','raw1260','raw_shift'] for c in [.01,.1,1.,10.] for e in [.01,.05,.09]]
    for k,c in enumerate(configs):
        X=bank[c['feature']];pred=np.full(y.shape,np.nan)
        for f in range(4):
            tr=m.fold!=f;te=(m.fold==f)&new
            model=make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(kernel='linear',C=c['C'],epsilon=c['epsilon'],max_iter=20000)))
            model.fit(X[tr],y[tr]);pred[te]=model.predict(X[te])
        rows.append(dict(candidate=k,**c,**metric(y[new],pred[new])))
        p=m.loc[new,['index','file','Y','T','scope','fold']].copy();p['candidate']=k;p['predY']=pred[new,0];p['predT']=pred[new,1];preds.append(p)
        if k%6==0:print(k,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(H/'robust_screen_summary.csv',index=False);pd.concat(preds).to_csv(H/'robust_screen_predictions.csv',index=False)
    print(pd.DataFrame(rows).sort_values(['passed','Y_max'],ascending=[False,True]).head(8).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()

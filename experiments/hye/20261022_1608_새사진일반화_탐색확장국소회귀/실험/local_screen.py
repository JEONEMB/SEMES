import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from data import H,load,features,metric
from forward_model import ForwardGLS

def main():
    m,a=load();bank=features(a);y=m[['Y','T']].to_numpy();new=m.dataset==20261020;h=a['D'][:,:,0].mean(axis=1)
    configs=[dict(feature=f,width=w,shrink=s) for f in ['raw1260','raw_shift'] for w in [20.,50.,100.] for s in [.01,.1]]
    rows=[];preds=[]
    for k,c in enumerate(configs):
        pred=np.full(y.shape,np.nan);X=bank[c['feature']]
        for i in np.flatnonzero(new):
            tr=m.fold!=m.fold[i];w=.01+np.exp(-.5*((h[tr]-h[i])/c['width'])**2)
            model=ForwardGLS(1,c['shrink']).fit(X[tr],y[tr],w)
            pred[i]=model.predict(X[i])[0]
        rows.append(dict(candidate=k,**c,**metric(y[new],pred[new])))
        p=m.loc[new,['index','file','Y','T','scope','fold']].copy();p['candidate']=k;p['predY']=pred[new,0];p['predT']=pred[new,1];preds.append(p)
        print(rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(H/'local_screen_summary.csv',index=False);pd.concat(preds).to_csv(H/'local_screen_predictions.csv',index=False)

if __name__=='__main__':
    with threadpool_limits(1):main()

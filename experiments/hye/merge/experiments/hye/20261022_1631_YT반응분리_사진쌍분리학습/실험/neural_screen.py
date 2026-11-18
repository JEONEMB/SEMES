from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;PREV=H.parent/'joint_yt_generalization_20261022'
sys.path.insert(0,str(PREV))
from data import load,features,metric
from neural_model import PairResidualNet

def main():
    m,a=load();X=features(a)['raw_shift'];y=m[['Y','T']].to_numpy();new=m.dataset.to_numpy()==20261020
    configs=[dict(components=k,pair_weight=w,crossfit=cf) for k in [16,64] for w in [0.,1.] for cf in [False,True]]
    records=[];preds=[];training=[]
    for k,c in enumerate(configs):
        p=np.full(y.shape,np.nan)
        for f in range(4):
            tr=m.fold.to_numpy()!=f;te=~tr
            model=PairResidualNet(c).fit(X[tr],y[tr],m.dataset.to_numpy()[tr],m.cv_group.to_numpy()[tr]);p[te]=model.predict(X[te]);training.append(dict(candidate=k,fold=f,losses=model.losses,device=model.device))
        for subset,mask in [('new60',new),('old281',~new),('all341',np.ones(len(y),bool))]:records.append(dict(candidate=k,subset=subset,**c,**metric(y[mask],p[mask])))
        pp=m[['index','dataset','file','Y','T','scope','condition','fold']].copy();pp['candidate']=k;pp['predY']=p[:,0];pp['predT']=p[:,1];preds.append(pp)
        print(k,records[-3],flush=True)
        pd.DataFrame(records).to_csv(H/'neural_summary.csv',index=False);pd.concat(preds).to_csv(H/'neural_predictions.csv',index=False)
    (H/'neural_protocol.json').write_text(json.dumps(dict(configs=configs,steps=800,learning_rate=.002,weight_decay=.05,pca='fit only training fold, whitening',base='raw_shift ForwardGLS .01',pair_training='sameT and sameY graph with targets; no test labels',grouped=True,training=training,scope='development comparison; conditional-label input forbidden'),indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()

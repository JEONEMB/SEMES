"""Development screen: fixed grouped folds, no filename features or test fitting."""
from pathlib import Path
import sys,json,time,os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'joint_center_630_20261022'))
sys.path.insert(0,str(ROOT/'codex_experiments_20261014/joint_yt_improved_20261022'))
from features import FeatureMaps
from fullfield import JointLinear,field_features

def metrics(y,p):
 e=abs(y-p);return dict(n=len(e),passed=int(np.all(e<=.09+1e-12,axis=1).sum()),Y_MAE=float(e[:,0].mean()),Y_max=float(e[:,0].max()),T_MAE=float(e[:,1].mean()),T_max=float(e[:,1].max()))

def make(c):
 f=c['family']
 if f=='ridge':return make_pipeline(StandardScaler(),Ridge(alpha=c['alpha']))
 if f=='pls':return make_pipeline(StandardScaler(),PLSRegression(n_components=c['components'],scale=False,max_iter=500))
 if f=='gls':return JointLinear(dict(family='gls_full',alpha=c['alpha']))
 if f=='poly2':return make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=c['alpha']))
 if f=='rbf':return make_pipeline(StandardScaler(),KernelRidge(alpha=c['alpha'],kernel='rbf',gamma=c['gamma']))
 if f=='svr':return make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(C=c['C'],epsilon=.005,gamma=c['gamma'])))

def main():
 data=ROOT/'joint_center_630_20261022/results';m=pd.read_csv(data/'manifest.csv');a=np.load(data/'measurements.npz');maps=FeatureMaps(a['points'],a['regions'])
 bank={f:maps.transform(a['D'],f) for f in ['P12','QR18','QR36','raw1260']}
 for f in ['grid4','grid8']:bank[f]=np.stack([field_features(a['points'],d,q,f) for d,q in zip(a['D'],a['quality'])])
 splits=json.loads((data/'expanded_features/splits.json').read_text());fold=np.full(len(m),-1)
 for f,s in enumerate(splits):fold[m.relative.isin(s['test'])]=f
 assert np.all(fold>=0)
 for _,g in m[m.scope=='mixed'].groupby('condition'):assert len(set(fold[g.index]))==1
 new=m.relative.str.startswith('20261020').to_numpy();old=~new;y=m[['Y','T']].to_numpy()
 cfg=[]
 for feat,X in bank.items():
  cfg += [dict(feature=feat,family='ridge',alpha=alpha) for alpha in [.0001,.01,1.,100.]]
  cfg += [dict(feature=feat,family='pls',components=k) for k in [2,4,8,16,32] if k<min(X.shape[1],38)]
  cfg += [dict(feature=feat,family='gls',alpha=alpha) for alpha in [.001,.01,.1,1.]]
  if feat in ['P12','QR18','QR36']:
   cfg += [dict(feature=feat,family='poly2',alpha=alpha) for alpha in [.01,1.,100.]]
  cfg += [dict(feature=feat,family='rbf',alpha=alpha,gamma=gamma/X.shape[1]) for alpha in [.001,.1] for gamma in [.1,1.]]
  cfg += [dict(feature=feat,family='svr',C=10.,gamma=gamma/X.shape[1]) for gamma in [.1,1.]]
 (H/'screen_protocol.json').write_text(json.dumps(dict(source=str(data),folds=str(data/'expanded_features/splits.json'),label_use='training and evaluation only',same_condition_grouped=True,scope='development comparison of held-out conditions, not independent after screening',configs=cfg),indent=2))
 records=[];allpred=[]
 for regime in ['new_only','combined','old_transfer']:
  for k,c in enumerate(cfg):
   started=time.monotonic();X=bank[c['feature']];pred=np.full(y.shape,np.nan);err=None
   try:
    if regime=='old_transfer':pred[new]=make(c).fit(X[old],y[old]).predict(X[new])
    else:
     for f in range(4):
      tr=(fold!=f)&(new if regime=='new_only' else np.ones(len(m),bool));te=(fold==f)&new
      if c['family']=='pls' and c['components']>=tr.sum():raise ValueError('too many PLS components')
      pred[te]=make(c).fit(X[tr],y[tr]).predict(X[te])
    scores=metrics(y[new],pred[new]);records.append(dict(regime=regime,candidate=k,**c,**scores,seconds=time.monotonic()-started))
    pp=m.loc[new,['index','file','relative','Y','T','scope','condition']].copy();pp['regime']=regime;pp['candidate']=k;pp['fold']=fold[new];pp['predY']=pred[new,0];pp['predT']=pred[new,1];allpred.append(pp)
   except Exception as exc:err=str(exc);records.append(dict(regime=regime,candidate=k,**c,error=err))
   if k%10==0:
    df=pd.DataFrame(records);df.to_csv(H/'screen_summary.csv',index=False);print(regime,k,c,scores if err is None else err,flush=True)
  df=pd.DataFrame(records);df.to_csv(H/'screen_summary.csv',index=False);pd.concat(allpred,ignore_index=True).to_csv(H/'screen_predictions.csv',index=False)
  print('BEST',regime,df[df.regime==regime].sort_values(['passed','Y_max'],ascending=[False,True]).head(8).to_string(index=False),flush=True)

if __name__=='__main__':
 with threadpool_limits(1):main()

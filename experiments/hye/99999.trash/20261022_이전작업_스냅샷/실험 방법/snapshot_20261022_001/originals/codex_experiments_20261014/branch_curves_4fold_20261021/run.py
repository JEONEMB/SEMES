"""Independent candidate slopes/curvature, train-only six-branch calibration."""
from pathlib import Path
import sys,json,itertools
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B),str(B/'multiref_4fold_20261021')]
import zero_experiment as Z
from experiment import enrich,stats
names=np.load(B/'center_patch_v2_20261021/tracks.npz')['names'];m=Z.manifest(names);ix={n:i for i,n in enumerate(names)};ids=np.flatnonzero(m.axis!='C')
features={'patch':np.load(B/'center_patch_v2_20261021/v2_patch630_features.npy'),'quality':np.load(B/'center_patch_v2_20261021/v2_patch630_quality_features.npy'),'gn31':np.load(B/'patch_windows_20261021/gn31_sigma0.8_features.npy'),'huber6':np.load(B/'robust_gn_20261021/huber6_features.npy')}
configs=list(itertools.product(features,[False,True],[.001,.05,.2,1.]))
def fit(config,tr):
    feature,quad,alpha=config;X=features[feature];branches={};residual=[];rank=0
    for k in np.unique(m.session.iloc[tr]):
        ii=tr[m.session.to_numpy()[tr]==k];v=m.value.to_numpy()[ii];design=np.c_[np.ones(len(v)),v,v*v] if quad else np.c_[np.ones(len(v)),v]
        coef=np.linalg.lstsq(design,X[ii],rcond=None)[0];rank+=np.linalg.matrix_rank(design);residual.append(X[ii]-design@coef)
        branches[k]=(k[0],coef[0],coef[1],coef[2] if quad else np.zeros(X.shape[1]))
    R=np.vstack(residual);S=R.T@R/(len(tr)-rank);std=np.sqrt(np.maximum(np.diag(S),1e-14));C=S/std[:,None]/std[None,:];C=(1-alpha)*C+alpha*np.eye(len(std));W=np.linalg.cholesky(np.linalg.inv(C)).T/std[None,:]
    return branches,W
def predict(config,tr,te,save=None):
    model=fit(config,tr)
    if save:
        branches,W=model;(H/save).write_text(json.dumps(dict(W=W.tolist(),branches={k:[v[0],*[x.tolist() for x in v[1:]]] for k,v in branches.items()})))
    return enrich(Z.predict(m,features[config[0]],te,model,'inferred_session'))
folds=json.loads((B/'multiref_4fold_20261021/extended/splits.json').read_text());(H/'splits.json').write_text(json.dumps(folds,indent=2));mapping={f'b{k:03d}':dict(feature=c[0],quadratic=c[1],shrinkage=c[2],curve_sharing='none') for k,c in enumerate(configs)};(H/'configs.json').write_text(json.dumps(mapping,indent=2))
rows=[];nested=[];selections=[]
for f,s in enumerate(folds):
    tr=np.array([ix[n] for n in s['train']]);te=np.array([ix[n] for n in s['test']]);inner=list(StratifiedKFold(3,shuffle=True,random_state=100+f).split(tr,m.session.iloc[tr]));rank=[];local=[]
    for k,c in enumerate(configs):
        ip=pd.concat([predict(c,tr[a],tr[b]) for a,b in inner]);ss=stats(ip);rank.append((ss['failures'],max(ss['Y_max'],ss['T_max']),k));local.append(predict(c,tr,te).assign(config=f'b{k:03d}',protocol='oof4',fold=f))
    chosen=min(rank)[2];rows+=local;nested.append(local[chosen].assign(config='nested_selected'));selections.append(dict(fold=f,config=f'b{chosen:03d}',rank=sorted(rank)));print('fold',f,'chosen',chosen,flush=True)
for k,c in enumerate(configs):rows.append(predict(c,ids,ids,save=f'b{k:03d}_model.json').assign(config=f'b{k:03d}',protocol='resub280',fold=-1))
p=pd.concat(rows+nested,ignore_index=True);p.to_csv(H/'predictions.csv',index=False);p[~p['pass']].to_csv(H/'failures.csv',index=False)
s=pd.DataFrame([dict(config=k,protocol=t,**mapping.get(k,{}),**stats(g)) for (k,t),g in p.groupby(['config','protocol'])]);s.to_csv(H/'summary.csv',index=False)
pd.DataFrame([dict(config=k,fold=f,**stats(g)) for (k,f),g in p[p.protocol=='oof4'].groupby(['config','fold'])]).to_csv(H/'fold_results.csv',index=False)
(H/'nested_selection.json').write_text(json.dumps(selections,indent=2));print(s[s.protocol=='oof4'].sort_values(['failures','Y_max']).head(10).to_string(index=False));print(s[s.config=='nested_selected'].to_string(index=False))

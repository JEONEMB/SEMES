"""Training-only heteroscedastic Gaussian branch likelihood; no test labels in inference."""
from pathlib import Path
import sys,json,itertools
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B),str(B/'multiref_4fold_20261021')]
import zero_experiment as Z
from experiment import enrich,stats
names=np.load(B/'center_patch_v2_20261021/tracks.npz')['names'];m=Z.manifest(names);ix={n:i for i,n in enumerate(names)};ids=np.flatnonzero(m.axis!='C')
features={'patch':np.load(B/'center_patch_v2_20261021/v2_patch630_features.npy'),'quality':np.load(B/'center_patch_v2_20261021/v2_patch630_quality_features.npy'),'gn31':np.load(B/'patch_windows_20261021/gn31_sigma0.8_features.npy')}
configs=list(itertools.product(features,['pooled','axis','session'],[.001,.2,1.],['uniform_branch','equal_axis','empirical']))
def fit(config,tr):
    feature,group,alpha,prior=config;X=features[feature];branches,_=Z.fit(m,X,tr,m.value.to_numpy(),'inferred_session',True)
    residual=[]
    for i in tr:
        _,b,s,c=branches[m.session.iloc[i]];v=m.value.iloc[i];residual.append(X[i]-b-s*v-c*v*v)
    residual=np.array(residual);S=residual.T@residual/len(tr);std=np.sqrt(np.maximum(np.diag(S),1e-14));r=residual/std;pooled=r.T@r/len(tr)
    models={}
    for k,(a,b,s,c) in branches.items():
        mask=np.ones(len(tr),bool) if group=='pooled' else (m.axis.to_numpy()[tr]==a if group=='axis' else m.session.to_numpy()[tr]==k)
        local=r[mask].T@r[mask]/sum(mask);cov=(1-alpha)*local+alpha*pooled;cov=.95*cov+.05*np.diag(np.diag(pooled));cov+=np.eye(len(std))*1e-9
        W=np.linalg.cholesky(np.linalg.inv(cov)).T/std[None,:];logdet=float(np.linalg.slogdet(cov)[1])
        prob=1/len(branches) if prior=='uniform_branch' else .5/sum(v[0]==a for v in branches.values()) if prior=='equal_axis' else np.mean(m.session.to_numpy()[tr]==k)
        models[k]=(a,b,s,c,W,logdet-2*np.log(prob))
    return models
def predict(config,tr,te,save=None):
    models=fit(config,tr);X=features[config[0]];choices=[]
    if save:(H/save).write_text(json.dumps({k:[v[0],*[x.tolist() if isinstance(x,np.ndarray) else float(x) for x in v[1:]]] for k,v in models.items()}))
    for k,(a,b,s,c,W,penalty) in models.items():
        z=(X[te]-b)@W.T;s=s@W.T;c=c@W.T;v=z@s/(s@s)
        for _ in range(8):
            j=s+2*v[:,None]*c;res=z-v[:,None]*s-v[:,None]**2*c;v+=np.sum(j*res,axis=1)/np.maximum(np.sum(j*j,axis=1),1e-20)
        cost=np.sum((z-v[:,None]*s-v[:,None]**2*c)**2,axis=1)+penalty
        cost=np.where(v*(1 if k.endswith('+') else -1)>=0,cost,np.inf);choices.append((k,a,v,cost))
    best=np.argmin(np.stack([r[3] for r in choices],axis=1),axis=1);rows=[]
    for t,i in enumerate(te):
        k,a,v,cost=choices[best[t]];rows.append(dict(file=m.file.iloc[i],axis=m.axis.iloc[i],value=m.value.iloc[i],session=m.session.iloc[i],pred_axis=a,pred=v[t],branch=k,cost=cost[t]))
    return enrich(pd.DataFrame(rows))
folds=json.loads((B/'multiref_4fold_20261021/extended/splits.json').read_text());(H/'splits.json').write_text(json.dumps(folds,indent=2));mapping={f'l{k:03d}':dict(feature=c[0],covariance_group=c[1],pool_fraction=c[2],prior=c[3]) for k,c in enumerate(configs)};(H/'configs.json').write_text(json.dumps(mapping,indent=2))
rows=[];nested=[];selections=[]
for f,s in enumerate(folds):
    tr=np.array([ix[n] for n in s['train']]);te=np.array([ix[n] for n in s['test']]);inner=list(StratifiedKFold(3,shuffle=True,random_state=100+f).split(tr,m.session.iloc[tr]));rank=[];local=[]
    for k,c in enumerate(configs):
        ip=pd.concat([predict(c,tr[a],tr[b]) for a,b in inner]);ss=stats(ip);rank.append((ss['failures'],max(ss['Y_max'],ss['T_max']),k));local.append(predict(c,tr,te).assign(config=f'l{k:03d}',protocol='oof4',fold=f))
    chosen=min(rank)[2];rows+=local;nested.append(local[chosen].assign(config='nested_selected'));selections.append(dict(fold=f,config=f'l{chosen:03d}',rank=sorted(rank)));print('fold',f,'chosen',chosen,flush=True)
for k,c in enumerate(configs):rows.append(predict(c,ids,ids,save=f'l{k:03d}_model.json').assign(config=f'l{k:03d}',protocol='resub280',fold=-1))
p=pd.concat(rows+nested,ignore_index=True);p.to_csv(H/'predictions.csv',index=False);p[~p['pass']].to_csv(H/'failures.csv',index=False)
s=pd.DataFrame([dict(config=k,protocol=t,**mapping.get(k,{}),**stats(g)) for (k,t),g in p.groupby(['config','protocol'])]);s.to_csv(H/'summary.csv',index=False)
pd.DataFrame([dict(config=k,fold=f,**stats(g)) for (k,f),g in p[p.protocol=='oof4'].groupby(['config','fold'])]).to_csv(H/'fold_results.csv',index=False)
(H/'nested_selection.json').write_text(json.dumps(selections,indent=2));print(s[s.protocol=='oof4'].sort_values(['failures','Y_max']).head(10).to_string(index=False));print(s[s.config=='nested_selected'].to_string(index=False))

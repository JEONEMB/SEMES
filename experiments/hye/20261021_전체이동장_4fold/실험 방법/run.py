"""Full630 displacement field, residual-SVD shrinkage, training-only calibration."""
from pathlib import Path
import sys,json,itertools
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B),str(B/'multiref_4fold_20261021')]
import zero_experiment as Z
from experiment import enrich,stats
base=np.load(B/'center_patch_v2_20261021/tracks.npz');names=base['names'];m=Z.manifest(names);ix={n:i for i,n in enumerate(names)};ids=np.flatnonzero(m.axis!='C')
features={}
for key,path in [('gn17',B/'center_patch_v2_20261021/tracks.npz'),('gn31',B/'patch_windows_20261021/gn31_sigma0.8_tracks.npz'),('huber6',B/'robust_gn_20261021/huber6_tracks.npz')]:
    a=np.load(path);assert np.array_equal(a['names'],names);D=a['D'].astype(float);D-=D[0];features[key]=D.reshape(len(D),-1)
configs=list(itertools.product(features,[False,True],[.01,.05,.2,.5,1.]))
def fit(config,tr):
    feature,quad,alpha=config;X=features[feature];branches={};residual=[]
    for axis in ['Y','T']:
        ii=tr[m.axis.to_numpy()[tr]==axis];v=m.value.to_numpy()[ii];ss=m.session.to_numpy()[ii];sessions=np.unique(ss)
        basis=np.c_[v,v*v] if quad else v[:,None];design=np.c_[basis,(ss[:,None]==sessions).astype(float)];coef=np.linalg.lstsq(design,X[ii],rcond=None)[0]
        residual.append(X[ii]-design@coef)
        for j,k in enumerate(sessions):branches[k]=(axis,coef[len(basis[0])+j],coef[0],coef[1] if quad else np.zeros(X.shape[1]))
    R=np.vstack(residual);std=np.sqrt(np.maximum(np.mean(R*R,axis=0),1e-10));R=R/std/np.sqrt(len(R))
    _,s,V=np.linalg.svd(R,full_matrices=False)
    adjust=1/np.sqrt(alpha+(1-alpha)*s*s)-1/np.sqrt(alpha)
    def whiten(x):
        z=x/std;return z/np.sqrt(alpha)+(z@V.T*adjust)@V
    branches={k:(a,whiten(b),whiten(s),whiten(c)) for k,(a,b,s,c) in branches.items()}
    return branches,whiten,dict(std=std,V=V,adjust=adjust,alpha=alpha)
def predict(config,tr,te,save=None):
    branches,whiten,params=fit(config,tr);X=whiten(features[config[0]][te]);choices=[]
    if save:
        np.savez_compressed(H/save,**params,branches=np.stack([np.stack(v[1:]) for v in branches.values()]),keys=np.array(list(branches)),axes=np.array([v[0] for v in branches.values()]))
    for k,(a,b,s,c) in branches.items():
        z=X-b;v=z@s/(s@s)
        for _ in range(8):
            j=s+2*v[:,None]*c;res=z-v[:,None]*s-v[:,None]**2*c;v+=np.sum(j*res,axis=1)/np.maximum(np.sum(j*j,axis=1),1e-20)
        cost=np.sum((z-v[:,None]*s-v[:,None]**2*c)**2,axis=1);cost=np.where(v*(1 if k.endswith('+') else -1)>=0,cost,np.inf);choices.append((k,a,v,cost))
    best=np.argmin(np.stack([r[3] for r in choices],axis=1),axis=1);rows=[]
    for t,i in enumerate(te):
        k,a,v,cost=choices[best[t]];rows.append(dict(file=m.file.iloc[i],axis=m.axis.iloc[i],value=m.value.iloc[i],session=m.session.iloc[i],pred_axis=a,pred=v[t],branch=k,cost=cost[t]))
    return enrich(pd.DataFrame(rows))
def main():
    folds=json.loads((B/'multiref_4fold_20261021/extended/splits.json').read_text());(H/'splits.json').write_text(json.dumps(folds,indent=2));mapping={f'f{k:03d}':dict(feature=c[0],quadratic=c[1],shrinkage=c[2],dimensions=1260) for k,c in enumerate(configs)};(H/'configs.json').write_text(json.dumps(mapping,indent=2))
    rows=[];nested=[];selections=[]
    for f,s in enumerate(folds):
        tr=np.array([ix[n] for n in s['train']]);te=np.array([ix[n] for n in s['test']]);inner=list(StratifiedKFold(3,shuffle=True,random_state=100+f).split(tr,m.session.iloc[tr]));rank=[];local=[]
        for k,c in enumerate(configs):
            ip=pd.concat([predict(c,tr[a],tr[b]) for a,b in inner]);ss=stats(ip);rank.append((ss['failures'],max(ss['Y_max'],ss['T_max']),k));local.append(predict(c,tr,te).assign(config=f'f{k:03d}',protocol='oof4',fold=f))
        chosen=min(rank)[2];rows+=local;nested.append(local[chosen].assign(config='nested_selected'));selections.append(dict(fold=f,config=f'f{chosen:03d}',rank=sorted(rank)));print('fold',f,'chosen',chosen,flush=True)
    for k,c in enumerate(configs):rows.append(predict(c,ids,ids,save=f'f{k:03d}_model.npz').assign(config=f'f{k:03d}',protocol='resub280',fold=-1))
    p=pd.concat(rows+nested,ignore_index=True);p.to_csv(H/'predictions.csv',index=False);p[~p['pass']].to_csv(H/'failures.csv',index=False)
    s=pd.DataFrame([dict(config=k,protocol=t,**mapping.get(k,{}),**stats(g)) for (k,t),g in p.groupby(['config','protocol'])]);s.to_csv(H/'summary.csv',index=False)
    pd.DataFrame([dict(config=k,fold=f,**stats(g)) for (k,f),g in p[p.protocol=='oof4'].groupby(['config','fold'])]).to_csv(H/'fold_results.csv',index=False)
    (H/'nested_selection.json').write_text(json.dumps(selections,indent=2));print(s[s.protocol=='oof4'].sort_values(['failures','Y_max']).head(10).to_string(index=False));print(s[s.config=='nested_selected'].to_string(index=False))
if __name__=='__main__':main()

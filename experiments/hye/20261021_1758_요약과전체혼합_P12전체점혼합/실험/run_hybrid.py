from pathlib import Path
import sys,json,itertools,functools
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B/'fullfield_4fold_20261021'),str(B/'multiref_4fold_20261021')]
import run as F
from experiment import fit,enrich,stats
m=F.m;ids=F.ids;ix=F.ix
X12={'gn17':np.load(B/'center_patch_v2_20261021/v2_patch630_features.npy'),'gn31':np.load(B/'patch_windows_20261021/gn31_sigma0.8_features.npy')}
configs=list(itertools.product(X12,[.2,.5,1.],[.03,.1,.3,1.,3.]))
@functools.lru_cache(maxsize=12)
def components(feature,alpha,train):
    tr=np.array(train);bb,white,_=F.fit((feature,True,alpha),tr);p,W=fit(X12[feature],m,tr,'inferred_session',True,.2)
    return bb,white(F.features[feature]),{k:(a,W@b,W@s,W@c) for k,(a,b,s,c) in p.items()},X12[feature]@W.T
def predict(config,tr,te):
    feature,alpha,beta=config;bb,XX,p,PP=components(feature,alpha,tuple(tr));weight=np.sqrt(beta*12/1260);choices=[]
    for k,(a,b,s,c) in p.items():
        _,bf,sf,cf=bb[k];z=np.c_[PP[te]-b,(XX[te]-bf)*weight];s=np.r_[s,sf*weight];c=np.r_[c,cf*weight];v=z@s/(s@s)
        for _ in range(8):
            j=s+2*v[:,None]*c;res=z-v[:,None]*s-v[:,None]**2*c;v+=np.sum(j*res,axis=1)/np.maximum(np.sum(j*j,axis=1),1e-20)
        cost=np.sum((z-v[:,None]*s-v[:,None]**2*c)**2,axis=1);cost=np.where(v*(1 if k.endswith('+') else -1)>=0,cost,np.inf);choices.append((k,a,v,cost))
    best=np.argmin(np.stack([r[3] for r in choices],axis=1),axis=1);rows=[]
    for t,i in enumerate(te):
        k,a,v,cost=choices[best[t]];rows.append(dict(file=m.file.iloc[i],axis=m.axis.iloc[i],value=m.value.iloc[i],session=m.session.iloc[i],pred_axis=a,pred=v[t],branch=k,cost=cost[t]))
    return enrich(pd.DataFrame(rows))
folds=json.loads((B/'multiref_4fold_20261021/extended/splits.json').read_text());(H/'splits.json').write_text(json.dumps(folds,indent=2));mapping={f'h{k:03d}':dict(feature=c[0],fullfield_shrinkage=c[1],fullfield_relative_weight=c[2],normalization='12/1260',P12_shrinkage=.2,quadratic=True) for k,c in enumerate(configs)};(H/'configs.json').write_text(json.dumps(mapping,indent=2))
rows=[];nested=[];selections=[]
for f,s in enumerate(folds):
    tr=np.array([ix[n] for n in s['train']]);te=np.array([ix[n] for n in s['test']]);inner=list(StratifiedKFold(3,shuffle=True,random_state=100+f).split(tr,m.session.iloc[tr]));rank=[];local=[]
    for k,c in enumerate(configs):
        ip=pd.concat([predict(c,tr[a],tr[b]) for a,b in inner]);ss=stats(ip);rank.append((ss['failures'],max(ss['Y_max'],ss['T_max']),k));local.append(predict(c,tr,te).assign(config=f'h{k:03d}',protocol='oof4',fold=f))
    chosen=min(rank)[2];rows+=local;nested.append(local[chosen].assign(config='nested_selected'));selections.append(dict(fold=f,config=f'h{chosen:03d}',rank=sorted(rank)));print('fold',f,'chosen',chosen,flush=True)
for k,c in enumerate(configs):rows.append(predict(c,ids,ids).assign(config=f'h{k:03d}',protocol='resub280',fold=-1))
p=pd.concat(rows+nested,ignore_index=True);p.to_csv(H/'predictions.csv',index=False);p[~p['pass']].to_csv(H/'failures.csv',index=False)
s=pd.DataFrame([dict(config=k,protocol=t,**mapping.get(k,{}),**stats(g)) for (k,t),g in p.groupby(['config','protocol'])]);s.to_csv(H/'summary.csv',index=False)
pd.DataFrame([dict(config=k,fold=f,**stats(g)) for (k,f),g in p[p.protocol=='oof4'].groupby(['config','fold'])]).to_csv(H/'fold_results.csv',index=False)
(H/'nested_selection.json').write_text(json.dumps(selections,indent=2));print(s[s.protocol=='oof4'].sort_values(['failures','Y_max']).head(10).to_string(index=False));print(s[s.config=='nested_selected'].to_string(index=False))

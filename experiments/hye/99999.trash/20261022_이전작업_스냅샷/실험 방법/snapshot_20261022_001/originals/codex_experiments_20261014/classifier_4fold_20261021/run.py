from pathlib import Path
import sys,json,itertools
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B),str(B/'multiref_4fold_20261021')]
import zero_experiment as Z
from experiment import fit,enrich,stats
names=np.load(B/'center_patch_v2_20261021/tracks.npz')['names'];m=Z.manifest(names);ix={n:i for i,n in enumerate(names)};ids=np.flatnonzero(m.axis!='C')
features={k:np.load(B/'center_patch_v2_20261021'/f'{k}_features.npy') for k in ['v1_edge630','v2_patch630','v2_patch630_quality']}
features['ecc_affine']=np.load(B/'multiref_4fold_20261021/ecc_affine_features.npy')
folds=json.loads((B/'multiref_4fold_20261021/extended/splits.json').read_text());(H/'splits.json').write_text(json.dumps(folds,indent=2))
settings=[('lda',s,0) for s in [.001,.05,.2,1.]]+[('svc',c,g) for c in [1.,10.,100.] for g in [.01,.1,1.]]
configs=list(itertools.product(features,['axis','session'],settings))
def predict(config,tr,te):
    feature,target,(kind,c,g)=config;X=features[feature]
    scaler=StandardScaler().fit(X[tr]);xt=scaler.transform(X[tr]);xe=scaler.transform(X[te])
    label=m.axis.to_numpy() if target=='axis' else m.session.to_numpy()
    clf=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=c) if kind=='lda' else SVC(C=c,gamma=g)
    clf.fit(xt,label[tr]);chosen=clf.predict(xe)
    branches,W=fit(X,m,tr,'inferred_session',True,.2);out=[]
    for group in np.unique(chosen):
        tt=te[chosen==group]
        allowed={k:v for k,v in branches.items() if (v[0]==group if target=='axis' else k==group)}
        # Axis classifier does not receive test labels; conditional magnitude is a curve projection.
        out.append(Z.predict(m,X,tt,(allowed,W),'inferred_session'))
    return enrich(pd.concat(out)).sort_values('file')
mapping={f'd{k:03d}':dict(feature=c[0],classifier_target=c[1],classifier=c[2][0],parameter=c[2][1],gamma=c[2][2]) for k,c in enumerate(configs)}
(H/'configs.json').write_text(json.dumps(mapping,indent=2));rows=[];nested=[];selections=[]
for f,s in enumerate(folds):
    tr=np.array([ix[n] for n in s['train']]);te=np.array([ix[n] for n in s['test']]);inner=list(StratifiedKFold(3,shuffle=True,random_state=100+f).split(tr,m.session.iloc[tr]));rank=[];local=[]
    for k,c in enumerate(configs):
        ip=pd.concat([predict(c,tr[a],tr[b]) for a,b in inner]);ss=stats(ip);rank.append((ss['failures'],max(ss['Y_max'],ss['T_max']),k))
        p=predict(c,tr,te).assign(config=f'd{k:03d}',protocol='oof4',fold=f);local.append(p)
    chosen=min(rank)[2];rows+=local;nested.append(local[chosen].assign(config='nested_selected'));selections.append(dict(fold=f,config=f'd{chosen:03d}',rank=sorted(rank)))
    print('fold',f,'selected',chosen,flush=True)
for k,c in enumerate(configs):rows.append(predict(c,ids,ids).assign(config=f'd{k:03d}',protocol='resub280',fold=-1))
p=pd.concat(rows+nested,ignore_index=True);p.to_csv(H/'predictions.csv',index=False);p[~p['pass']].to_csv(H/'failures.csv',index=False)
s=pd.DataFrame([dict(config=k,protocol=t,**mapping.get(k,{}),**stats(g)) for (k,t),g in p.groupby(['config','protocol'])]);s.to_csv(H/'summary.csv',index=False)
pd.DataFrame([dict(config=k,fold=f,**stats(g)) for (k,f),g in p[p.protocol=='oof4'].groupby(['config','fold'])]).to_csv(H/'fold_results.csv',index=False)
(H/'nested_selection.json').write_text(json.dumps(selections,indent=2));print(s[s.protocol=='oof4'].sort_values(['failures','Y_max']).head(10).to_string(index=False));print(s[s.config=='nested_selected'].to_string(index=False))

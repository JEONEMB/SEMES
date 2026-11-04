"""Verified fixed-630 observations and shared grouped validation splits."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
sys.path.insert(0,str(ROOT/'joint_center_630_20261022'))
from features import FeatureMaps

def load():
    a=np.load(H/'measurements/features.npz');m=pd.read_csv(H/'measurements/manifest.csv')
    src=pd.read_csv(ROOT/'joint_center_630_20261022/results/manifest.csv')
    original=json.loads((ROOT/'joint_center_630_20261022/results/expanded_features/splits.json').read_text())
    source_fold={name:f for f,s in enumerate(original) for name in s['test']}
    m['relative']=[src.iloc[int(i)].relative for i in m.expanded_cache_index]
    m['fold']=[source_fold[name] for name in m.relative]
    m.loc[(m.dataset==20261020)&(m.scope=='center'),'fold']=0
    m['cv_group']=np.where(m.dataset==20261020,'new:'+m.condition,'old:'+m.relative)
    for _,g in m.groupby('cv_group'):assert g.fold.nunique()==1
    assert len(m)==341 and m.success.all() and a['D'].shape==(341,630,2)
    return m,a

def features(a):
    maps=FeatureMaps(a['points'],a['regions']);d=a['D'];raw=d.reshape(len(d),-1)
    h=d[:,:,0].mean(axis=1,keepdims=True)/100
    bank={f:maps.transform(d,f) for f in ['P12','QR18','QR36']}
    bank.update(raw1260=raw,raw_shift=np.c_[raw,raw*h],raw_shift2=np.c_[raw,raw*h,raw*h*h])
    return bank

def metric(y,p):
    e=abs(np.asarray(y)-np.asarray(p));finite=np.isfinite(e).all(axis=1);e=e[finite]
    return dict(n=len(y),failures=int((~finite).sum()),passed=int(np.all(e<=.09+1e-12,axis=1).sum()),Y_MAE=float(e[:,0].mean()),Y_max=float(e[:,0].max()),T_MAE=float(e[:,1].mean()),T_max=float(e[:,1].max()))

def save_splits(m):
    rows=[dict(fold=f,train_indices=m.index[m.fold!=f].tolist(),test_indices=m.index[m.fold==f].tolist(),train=m.relative[m.fold!=f].tolist(),test=m.relative[m.fold==f].tolist()) for f in range(4)]
    (H/'splits.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    m.to_csv(H/'manifest_with_folds.csv',index=False,encoding='utf-8-sig')
    return rows

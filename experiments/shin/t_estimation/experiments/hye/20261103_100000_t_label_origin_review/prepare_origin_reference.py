"""Re-run the existing research origin model; native C++ evaluates T predictions.
No image/label is modified. Excluded rows never enter origin optimization.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
ROOT=P.parents[3]
sys.path.insert(0,str(ROOT/'analysis/frame_link_audit_20261103'))
from refine_origin import load,audited_data,RefinedModel

def fold(y,t):
    h=2166136261
    for c in f'{round(y*100)}:{round(t*100)}'.encode():h=((h^c)*16777619)&0xffffffff
    return h%5

def main():
    original,f,a=load();d=audited_data(original,True)
    excluded=set(pd.read_csv(P/'exclusions.csv').path)
    keep=~d.path.isin(excluded)
    assert keep.sum()==567 and len(excluded)==8
    assigned=np.array([fold(y,t) for y,t in d[['Y','T_original']].to_numpy()])
    refs=d.cohort.to_numpy()=='reference'
    jobs=[('full',keep.to_numpy())]
    for outer in range(5):
        jobs.append((f'outer{outer}',keep&((assigned!=outer)|refs)))
        for inner in range(5):
            if inner!=outer:jobs.append((f'outer{outer}_inner{inner}',keep&(((assigned!=outer)&(assigned!=inner))|refs)))
    (P/'origins').mkdir(exist_ok=True)
    for name,mask in jobs:
        dest=P/'origins'/f'{name}.csv'
        if dest.exists():continue
        train=d[np.asarray(mask)].copy()
        assert not set(train.path)&excluded
        model=RefinedModel(train,f,a,False,False,True).fit(maxiter=450)
        # Equal image weights: suspicious photos have been removed, not downweighted.
        sid=d.group.to_numpy(int)-1
        target=d[['Y','T']].to_numpy()+model.B[sid]
        export=pd.DataFrame({'path':d.path,'Y_standard':target[:,0],'T_standard':target[:,1],
                             'available':model.present[sid].astype(int)})
        export.to_csv(dest,index=False,float_format='%.17g')
        info={'train_paths':train.path.tolist(),'excluded_paths':sorted(excluded),'present':model.present.tolist(),
              'bias':model.B.tolist(),'raw_bound_hits':int((np.abs(model.u)>2.999).sum()),
              'sigma_y':model.sy,'sigma_t':model.st,'kind':'statistical coordinate convention; no physical ground truth'}
        dest.with_suffix('.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,'train',len(train),'bound_hits',info['raw_bound_hits'],flush=True)
    print('Completed26clean_origin_fits',flush=True)
if __name__=='__main__':main()

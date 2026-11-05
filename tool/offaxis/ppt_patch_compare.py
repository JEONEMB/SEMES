"""PPT slide 24-style patch sensitivity, with explicitly defined patch bounds.

The original PPT does not publish its old fixed-window coordinates. Therefore
the second ROI is specified as regenerated QR module extent +8px, not claimed
to be that unpublished original ROI. Both variants keep Gaussian kernel=1.
"""
import argparse,concurrent.futures,json
from pathlib import Path
import numpy as np
import pandas as pd
from extract import affine,load


def worker(task):
    root,rel,models=task
    image,base=load(Path(root)/rel),load(Path(root)/'center.bmp')
    rows=[]
    for qr in models:
        _,r=affine(image,base,qr)
        rows.append(dict(path=rel,qr=qr['slot'],patch='module_extent_plus8',**r))
    return rows


def main(root,previous,output):
    refs=json.loads((previous/'raw/reference.json').read_text());models=[]
    for q in refs:
        pts=np.array([f['reference'] for f in q['features']])
        left,top=np.floor(pts.min(axis=0)-8).astype(int)
        right,bottom=np.ceil(pts.max(axis=0)+8).astype(int)
        models.append(dict(slot=q['slot'],bounds=[int(left),int(top),int(right-left),int(bottom-top)],center=q['center']))
    inv=pd.read_csv(previous/'sampling.csv')
    paths=inv[inv.category.isin(['mixed','new-reference','reference'])].path.tolist()
    tasks=[(str(root),path,models) for path in paths]
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(worker,tasks):rows.extend(result)
    result=pd.DataFrame(rows)
    label=pd.read_csv(previous/'raw/ecc.csv');label=label[label.path.isin(paths)].copy();label['patch']='label_204x216'
    result=pd.concat([result,label],ignore_index=True)
    for patch,g in result.groupby('patch'):
        ref=g[g.path=='기준 위치/center1.bmp'].set_index('qr')
        for dim in ['dx','dy']:
            result.loc[g.index,'relative_'+dim]=g[dim]-g.qr.map(ref[dim])
    result.to_csv(output/'patch_comparison.csv',index=False)
    (output/'patch_bounds.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
    print(result.groupby('patch').correlation.agg(['min','median']).to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','previous','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();main(a.root,a.previous,a.output)

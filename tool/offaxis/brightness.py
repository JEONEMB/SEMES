"""Measure original RGB patch means at ECC-translated QR centers, without filtering."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def run(root, output):
    inventory=pd.read_csv(output/'sampling.csv')
    ecc=pd.read_csv(output/'raw/ecc.csv')
    paths=inventory[inventory.category.isin(['mixed','new-reference'])].path
    rows=[]
    for path,g in ecc[ecc.path.isin(paths)].groupby('path'):
        image=np.array(Image.open(root/path))
        for r in g.itertuples():
            if r.status!='ok':
                raise ValueError(f'ECC failed: {path}, {r.qr}')
            xx={'L':1644,'C':1902,'R':2160}[r.qr]
            yy={'L':1068,'C':1069,'R':1070}[r.qr]
            x,y=int(round(xx+r.dx)),int(round(yy+r.dy))
            crop=image[y-108:y+108,x-102:x+102]
            if crop.shape!=(216,204,3):
                raise ValueError(f'Unexpected RGB crop: {path}, {crop.shape}')
            rows.append(dict(path=path,qr=r.qr,mean_gray=float(crop[...,0].mean()),
                channels_equal=bool((crop[...,0]==crop[...,1]).all() and (crop[...,0]==crop[...,2]).all())))
    result=pd.DataFrame(rows).merge(inventory,on='path',validate='many_to_one')
    result.to_csv(output/'brightness.csv',index=False)
    print(f'{len(result)} original-pixel QR patch means')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)

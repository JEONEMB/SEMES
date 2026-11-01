"""Reconstruct the corner protocol documented on reference PPT slides 4 and 17.

Keep all 755 corners, including 125 diagonal crossings. Recenter each window
after fitting. PPT's diagnostic criteria do not reject a PSF lower-bound hit.
The unpublished original fitting code is unavailable; this is an explicit
reconstruction, not a claim of byte-identical original software.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
import qrcode
from qrcode.util import QRData
from extract import fit_corner,load


def reference(original):
    result=[]
    for qr in original:
        info=qr['qr_model']
        code=qrcode.QRCode(version=1,error_correction=info['ec'],border=0,mask_pattern=info['mask'])
        code.add_data(QRData(qr['payload'],mode=info['encoding']),optimize=0);code.make(fit=False)
        bits=np.pad(np.array(code.get_matrix(),dtype=int),1)
        model=np.float32([[f['col'],f['row']] for f in qr['features']])
        seeds=np.float32([f['seed'] for f in qr['features']])
        H,_=cv2.findHomography(model,seeds,0)
        features=[]
        for row in range(22):
            for col in range(22):
                b=bits[row:row+2,col:col+2]
                diagonal=b.sum()==2 and b[0,0]==b[1,1] and b[0,1]==b[1,0]
                if b.sum() not in [1,3] and not diagonal:continue
                pts=cv2.perspectiveTransform(np.float32([[[col,row],[col+1,row],[col,row+1]]]),H)[0]
                basis=(pts[1:]-pts[0]).T;basis/=np.linalg.norm(basis,axis=0)
                features.append(dict(id=row*22+col,row=row,col=col,bits=b.tolist(),seed=pts[0].tolist(),basis=basis.tolist(),diagonal=bool(diagonal)))
        result.append(dict(slot=qr['slot'],features=features))
    return result


def extract_frame(args):
    root,rel,models,ecc,old=args
    cv2.setNumThreads(1)
    image=load(Path(root)/rel);rows=[]
    for qr in models:
        slot=qr['slot'];r=ecc[slot]
        warp=np.array([[r['a00'],r['a01'],r['a02']],[r['a10'],r['a11'],r['a12']]])
        for f in qr['features']:
            projected=warp[:,:2]@f['seed']+warp[:,2]
            basis=warp[:,:2]@np.array(f['basis']);basis/=np.linalg.norm(basis,axis=0)
            cached=old.get((slot,f['id']))
            if cached is None:
                first=fit_corner(image,projected,f['bits'],basis,4)
                seed9=np.array([first['x'],first['y']])
            else:
                seed9=np.array(cached)
            # A fresh integer-pixel window centered at the first fitted point.
            small=fit_corner(image,seed9,f['bits'],basis,4)
            large_first=fit_corner(image,seed9,f['bits'],basis,5)
            large=fit_corner(image,[large_first['x'],large_first['y']],f['bits'],basis,5)
            window_difference=np.hypot(small['x']-large['x'],small['y']-large['y'])
            shift=np.hypot(small['x']-projected[0],small['y']-projected[1])
            flags=dict(converged=small['success'] and large['success'],rmse_ok=small['rmse']<=15,
                       contrast_ok=small['contrast']>=45,window_ok=window_difference<=.65,
                       shift_ok=shift<=3.6,sigma_ok=small['sigma']<=2.6)
            rows.append(dict(path=rel,qr=slot,id=f['id'],diagonal=f['diagonal'],
                projected_x=projected[0],projected_y=projected[1],window_difference=window_difference,
                projected_shift=shift,accepted=all(flags.values()),**flags,**small))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    models=reference(json.loads((a.previous/'raw/reference.json').read_text()))
    (a.output/'reference.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
    original=pd.read_csv(a.previous/'raw/points.csv')
    previous={path:{(r.qr,r.id):(r.x,r.y) for r in g.itertuples()} for path,g in original.groupby('path')}
    ec=pd.read_csv(a.previous/'raw/ecc.csv')
    transforms={path:{r['qr']:r for r in g.to_dict('records')} for path,g in ec.groupby('path')}
    inventory=pd.read_csv(a.previous/'sampling.csv')
    # New data first, followed by original sweeps. Every image is still retained.
    inventory['priority']=inventory.category.map({'reference':0,'new-reference':1,'mixed':2,'Y-only':3,'theta-only':4})
    paths=inventory.sort_values(['priority','path']).path.tolist()
    if a.limit:paths=paths[:a.limit]
    tasks=[(str(a.root),rel,models,transforms[rel],previous.get(rel,{})) for rel in paths]
    print('candidates',[(m['slot'],len(m['features'])) for m in models],flush=True)
    rows=[];start=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,result in enumerate(pool.map(extract_frame,tasks)):
            rows.extend(result)
            if (i+1)%10==0:print(f'{i+1}/{len(paths)} frames; {time.time()-start:.1f}s',flush=True)
    pd.DataFrame(rows).to_csv(a.output/'points.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    manifest=dict(frames=len(paths),points=len(rows),seconds=time.time()-start,theta_unit='degree',
        theta_unit_evidence='user confirmation 2026-10-20',input_smoothing=False,
        warm_start='previous raw 9x9 fitted L-corners; diagonal corners initialized from regenerated QR geometry',
        protocol='9x9 recentered fit; 11x11 fit then recentered fit; PPT diagnostic thresholds',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('extract.py')]})
    (a.output/'extraction.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()

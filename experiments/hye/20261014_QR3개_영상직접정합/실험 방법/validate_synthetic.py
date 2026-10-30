"""Known cubic subpixel motion plus photometric changes; tests dense optimizer."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,json
import numpy as np
from scipy import ndimage
from direct_pair import DenseRegistration,read_gray,basis

HERE=Path(__file__).resolve().parent
ROOT=Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
ref=read_gray(ROOT/'center.bmp'); model=DenseRegistration(ref,inset=10)
x0,y0,x1,y1=model.box
rawcrop=ref[y0:y1,x0:x1].astype(float)
yy,xx=np.mgrid[:rawcrop.shape[0],:rawcrop.shape[1]]
rng=np.random.default_rng(20261014)
rows=[]
motions=[(0.,0.),(.10,.07),(.25,-.30),(.5,.5),(7.7,-1.2),(28.35,.08),(80.15,-.45),(-79.4,.3)]
for noise,quantize in [(0.,False),(.6,True)]:
    for dx,dy in motions:
        gain,offset=1.025,2.3
        cur=ndimage.map_coordinates(rawcrop,[yy-dy,xx-dx],order=3,mode='reflect')*gain+offset
        if noise:cur+=rng.normal(0,noise,cur.shape)
        if quantize:cur=np.clip(np.rint(cur),0,255)
        canvas=ref.astype(float);canvas[y0:y1,x0:x1]=cur
        initial=np.zeros(12);initial[:2]=[dx+.15,dy-.10]
        r=model.refine(canvas,initial)
        truth=np.zeros(12);truth[:2]=[dx,dy]
        errs=np.concatenate([g[2]@(r['X']-truth).reshape(6,2) for g in model.geometry[1]])
        norms=np.linalg.norm(errs,axis=1)
        rows.append(dict(dx=dx,dy=dy,noise_std=noise,quantized=quantize,input_gain=gain,input_offset=offset,bias_dx=float(errs[:,0].mean()),bias_dy=float(errs[:,1].mean()),rmse_coordinate_px=float(np.sqrt((errs**2).mean())),p95_position_error_px=float(np.quantile(norms,.95)),max_position_error_px=float(norms.max()),mse_final=r['mse_final'],iterations=r['iterations']))
with (HERE/'synthetic_validation.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={"n_cases":len(rows),"max_coordinate_rmse_px":max(r['rmse_coordinate_px'] for r in rows),"max_p95_position_error_px":max(r['p95_position_error_px'] for r in rows),"max_position_error_px":max(r['max_position_error_px'] for r in rows),"note":"Controlled dense-optimizer test initialized within 0.18px of known warp; independent PairExtractor coarse+GN tested separately. This tests numerical tracking accuracy, not physical stage ground truth."}
(HERE/'synthetic_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary))

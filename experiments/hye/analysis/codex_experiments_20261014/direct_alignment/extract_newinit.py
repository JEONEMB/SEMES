"""Batch standalone dense pipeline and compare old-cache initialization only."""
import os
for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[k]="1"
from pathlib import Path
import argparse,time,json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from direct_pair import DirectPairExtractor,read_gray,basis

HERE=Path(__file__).resolve().parent
ROOT=Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")

def init():
    global MODEL,OLD
    MODEL=DirectPairExtractor(ROOT/"center.bmp",inset=10)
    z=np.load(HERE/"direct_p12.npz");OLD={str(n):x for n,x in zip(z['names'],z['initial'])}

def job(name):
    path=ROOT/name if name=="center.bmp" else ROOT/("Theta" if name.startswith("T") else "Y")/name
    out=MODEL.extract(path)
    comparison=MODEL.dense.refine(read_gray(path),OLD[name])
    return name,out,comparison

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    z=np.load(HERE/'direct_p12.npz');names=list(z['names']);names=names[:args.limit] if args.limit else names
    t0=time.time();out={}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init) as ex:
        fs=[ex.submit(job,str(n)) for n in names]
        for f in as_completed(fs):
            n,r,o=f.result();out[n]=(r,o)
            if len(out)%20==0 or len(out)==len(names):print(f"{len(out)}/{len(names)} elapsed={time.time()-t0:.1f}s",flush=True)
    new=[out[n][0] for n in names];old=[out[n][1] for n in names]
    arrays={k:np.asarray([r[k] for r in new]) for k in new[0]};arrays['names']=np.array(names)
    arrays['X_oldinit_same_support']=np.stack([r['X'] for r in old]);arrays['mse_oldinit_same_support']=np.array([r['mse_final'] for r in old])
    path=HERE/'direct_p12_newinit.npz';np.savez_compressed(path,**arrays)
    pts=np.array([[1563,982],[1721,1153],[1821,983],[1980,1154],[2079,984],[2239,1157],[1642,1067],[1900,1068],[2159,1070]])
    diff=np.einsum('pk,nkd->npd',basis(pts[:,0],pts[:,1]),(arrays['X']-arrays['X_oldinit_same_support']).reshape(-1,6,2))
    perimage=np.linalg.norm(diff,axis=2).max(axis=1)
    meta={'n_images':len(names),'pixel_count':int(new[0]['pixel_count']),'inset':10,'elapsed_seconds':time.time()-t0,'max_old_new_init_warp_difference_px':float(perimage.max()),'median_old_new_init_warp_difference_px':float(np.median(perimage)),'worst_name':str(names[int(perimage.argmax())]),'mse_mean_new':float(arrays['mse_final'].mean()),'mse_mean_old':float(arrays['mse_oldinit_same_support'].mean()),'initializer':'standalone PairExtractor GN17, QR only, no displacement cache','basis_layout':'[1,x,y,xx,xy,yy] interleaved dx,dy; x=(pixel_x-1906)/486 y=(pixel_y-1070)/122'}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print(json.dumps(meta),flush=True)

if __name__=='__main__':main()

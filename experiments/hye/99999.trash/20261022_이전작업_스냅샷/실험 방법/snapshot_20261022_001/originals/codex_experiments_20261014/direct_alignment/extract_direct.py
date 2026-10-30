"""QR-only dense photometric registration from center/current image pairs.

No file-name labels or session metadata enter alignment. Existing image-derived
point displacements initialize a global degree-2 warp. Template pixels lie
strictly inside the three QR rectangles. Per-QR brightness gain and offset are
profiled out on each iteration. Writes only under this directory.
"""
import os
for _k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "1"
from pathlib import Path
import time, json, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
Q = pd.read_csv(PROJ / "14_데이터_QR마커위치.csv").set_index("file")
NAMES = list(Q.index)
F = pd.read_csv(PROJ / "analysis_v6/features_qr_hp8.csv")
KEEP = F.region.str.startswith("QR").to_numpy()
F = F[KEEP]
Z = np.load(PROJ / "analysis_v6/tracks_qr_hp8.npz")
assert list(Z["names"]) == NAMES
D = Z["D_gn"][:, KEEP].astype(float)
D -= D[NAMES.index("center.bmp")]
XC, YC, XS, YS = 1906., 1070., 486., 122.

def basis(x, y):
    x, y = (x-XC)/XS, (y-YC)/YS
    return np.stack([np.ones_like(x),x,y,x*x,x*y,y*y],1)

INIT = np.linalg.lstsq(basis(F.x.to_numpy(), F.y.to_numpy()), D.transpose(1,0,2).reshape(len(F),-1),rcond=None)[0].reshape(6,len(D),2).transpose(1,0,2).reshape(len(D),12)
cv2.setNumThreads(1)

def read_image(name):
    p = ROOT/name if name == "center.bmp" else ROOT/("Theta" if name.startswith("T") else "Y")/name
    im = cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_GRAYSCALE)
    return im

def setup():
    global BOX, GEOM, TEMPLATE, PIXELS
    cen=Q.loc["center.bmp"]
    rects=[]
    for key in "LCR":
        cx,cy,w,h=[float(cen[f"{key}{k}"]) for k in ["x","y","w","h"]]
        rects.append((int(np.ceil(cx-w/2))+3,int(np.ceil(cy-h/2))+3,int(np.floor(cx+w/2))-3,int(np.floor(cy+h/2))-3))
    x0=min(r[0] for r in rects)-120; y0=min(r[1] for r in rects)-24
    x1=max(r[2] for r in rects)+121; y1=max(r[3] for r in rects)+25
    BOX=(x0,y0,x1,y1)
    crop=read_image("center.bmp")[y0:y1,x0:x1].astype(np.float64)
    TEMPLATE=cv2.GaussianBlur(crop,(0,0),1.0)
    GEOM={}
    for stride in [2,1]:
        regions=[]
        for ax,ay,bx,by in rects:
            yy,xx=np.mgrid[ay:by:stride,ax:bx:stride]
            x,y=xx.ravel(),yy.ravel()
            regions.append((x.astype(float),y.astype(float),basis(x,y),TEMPLATE[y-y0,x-x0]))
        GEOM[stride]=regions
    PIXELS=sum(len(g[0]) for g in GEOM[1])

def evaluate(coeff, p, stride, derivative=True):
    cI,cX,cY=coeff; x0,y0,_,_=BOX
    normal=np.zeros((12,12)); rhs=np.zeros(12); sse=0.; count=0; gains=[]
    P=p.reshape(6,2)
    for x,y,B,T in GEOM[stride]:
        delta=B@P
        coords=[y+delta[:,1]-y0,x+delta[:,0]-x0]
        I=ndimage.map_coordinates(cI,coords,order=3,prefilter=False,mode="nearest")
        IC=I-I.mean(); TC=T-T.mean(); a=float(IC@TC/max(IC@IC,1e-12)); b=float(T.mean()-a*I.mean())
        r=T-a*I-b; sse+=float(r@r); count+=len(T); gains.append([a,b])
        if derivative:
            Ix=ndimage.map_coordinates(cX,coords,order=3,prefilter=False,mode="nearest")*a
            Iy=ndimage.map_coordinates(cY,coords,order=3,prefilter=False,mode="nearest")*a
            J=np.empty((len(T),12)); J[:,0::2]=Ix[:,None]*B; J[:,1::2]=Iy[:,None]*B
            # Schur complement for per-QR brightness gain and constant offset.
            JC=J-J.mean(axis=0); v=JC.T@IC; denom=max(float(IC@IC),1e-12)
            normal+=JC.T@JC-np.outer(v,v)/denom
            rhs+=JC.T@r-v*float(IC@r)/denom
    return normal,rhs,sse/count,gains

def job(index):
    name=NAMES[index]; p=INIT[index].copy(); x0,y0,x1,y1=BOX
    if name=="center.bmp": return index,np.zeros(12),0.,0.,0,1.,[[1.,0.]]*3
    crop=read_image(name)[y0:y1,x0:x1].astype(np.float64)
    im=cv2.GaussianBlur(crop,(0,0),1.0)
    gy,gx=np.gradient(im)
    coeff=[ndimage.spline_filter(w,order=3) for w in (im,gx,gy)]
    start_mse=evaluate(coeff,p,1,False)[2]
    niter=0; condition=np.nan
    for stride,maxit in [(2,12),(1,10)]:
        for it in range(maxit):
            H,g,mse,gains=evaluate(coeff,p,stride)
            condition=float(np.linalg.cond(H)); dp=np.linalg.solve(H+np.eye(12)*1e-8,g)
            # Initial positions are already close. Limit proposed warp displacement.
            maxdelta=max(float(np.max(np.linalg.norm(geom[2]@dp.reshape(6,2),axis=1))) for geom in GEOM[stride])
            if maxdelta>0.5: dp*=0.5/maxdelta
            accepted=False
            for rate in [1.,0.5,0.25,0.125]:
                candidate=p+rate*dp
                newmse=evaluate(coeff,candidate,stride,False)[2]
                if newmse <= mse+1e-10:
                    p=candidate; accepted=True; break
            niter+=1
            if not accepted or maxdelta*rate < 2e-5: break
    final=evaluate(coeff,p,1,False)
    return index,p,start_mse,final[2],niter,condition,final[3]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=3); ap.add_argument("--limit",type=int,default=0); ap.add_argument("--output",default="direct_p12.npz"); args=ap.parse_args()
    setup(); limit=args.limit or len(NAMES); indices=list(range(limit)); out={}; t0=time.time()
    print(f"pixels={PIXELS} images={limit} workers={args.workers}",flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=setup) as pool:
        fs=[pool.submit(job,i) for i in indices]
        for ft in as_completed(fs):
            r=ft.result(); out[r[0]]=r
            if len(out)%10==0 or len(out)==limit: print(f"{len(out)}/{limit} elapsed={time.time()-t0:.1f}s",flush=True)
    ordered=[out[i] for i in indices]
    path=HERE/args.output
    np.savez_compressed(path,X=np.stack([r[1] for r in ordered]),names=np.array(NAMES[:limit]),initial=INIT[:limit],mse_initial=np.array([r[2] for r in ordered]),mse_final=np.array([r[3] for r in ordered]),iterations=np.array([r[4] for r in ordered]),condition=np.array([r[5] for r in ordered]),gains=np.array([r[6] for r in ordered]))
    meta={"pixel_count":PIXELS,"basis":["1","x","y","xx","xy","yy"],"layout":"basis-major interleaved dx,dy","x_center":XC,"y_center":YC,"x_scale":XS,"y_scale":YS,"sigma":1.,"qr_box_inset_pixels":3,"sampling":"scipy cubic spline, float64","brightness":"independent gain and bias per QR","elapsed_seconds":time.time()-t0,"n_images":limit,"mse_mean_initial":float(np.mean([r[2] for r in ordered])),"mse_mean_final":float(np.mean([r[3] for r in ordered]))}
    path.with_suffix(".json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(meta),flush=True)

if __name__=="__main__": main()

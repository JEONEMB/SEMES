"""Standalone center/current QR-only dense P12 measurement, no motion labels.

The fixed QR bounds are geometry configuration for this 3840x2160 camera.
The 10-pixel inset excludes plate/screw pixels from template objectives and
leaves room for sigma-1 blur, gradient, and cubic interpolation support.
"""
import os
for _k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_k] = "1"
from pathlib import Path
import sys, argparse, json, time
import cv2
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "measurement"))
from extract_pair import PairExtractor

QR_BOUNDS = [(1563,982,1721,1153),(1821,983,1980,1154),(2079,984,2239,1157)]
XC,YC,XS,YS = 1906.,1070.,486.,122.
cv2.setNumThreads(1)

def basis(x,y):
    x,y=(x-XC)/XS,(y-YC)/YS
    return np.stack([np.ones_like(x),x,y,x*x,x*y,y*y],1)

def read_gray(path):
    a=cv2.imdecode(np.fromfile(Path(path),np.uint8),cv2.IMREAD_GRAYSCALE)
    if a is None: raise ValueError(f"Cannot decode image: {path}")
    if a.shape != (2160,3840): raise ValueError(f"Expected fixed-camera 3840x2160 image, got {a.shape}")
    return a

class DenseRegistration:
    def __init__(self,reference,inset=10):
        self.inset=inset
        self.rects=[(a+inset,b+inset,c-inset,d-inset) for a,b,c,d in QR_BOUNDS]
        x0=min(r[0] for r in QR_BOUNDS)-120; y0=min(r[1] for r in QR_BOUNDS)-24
        x1=max(r[2] for r in QR_BOUNDS)+121; y1=max(r[3] for r in QR_BOUNDS)+25
        self.box=(x0,y0,x1,y1)
        self.reference=cv2.GaussianBlur(reference[y0:y1,x0:x1].astype(np.float64),(0,0),1.0)
        self.geometry={}
        for stride in [2,1]:
            regions=[]
            for a,b,c,d in self.rects:
                yy,xx=np.mgrid[b:d:stride,a:c:stride]; x,y=xx.ravel(),yy.ravel()
                regions.append((x.astype(float),y.astype(float),basis(x,y),self.reference[y-y0,x-x0]))
            self.geometry[stride]=regions
        self.pixel_count=sum(len(g[0]) for g in self.geometry[1])

    def coefficients(self,current):
        x0,y0,x1,y1=self.box
        im=cv2.GaussianBlur(current[y0:y1,x0:x1].astype(np.float64),(0,0),1.0)
        gy,gx=np.gradient(im)
        return [ndimage.spline_filter(w,order=3) for w in [im,gx,gy]]

    def evaluate(self,coeff,p,stride,derivative=True):
        cI,cX,cY=coeff; x0,y0,_,_=self.box
        normal=np.zeros((12,12)); rhs=np.zeros(12); sse=0.; count=0; gains=[]
        for x,y,B,T in self.geometry[stride]:
            delta=B@p.reshape(6,2); coords=[y+delta[:,1]-y0,x+delta[:,0]-x0]
            I=ndimage.map_coordinates(cI,coords,order=3,prefilter=False,mode="nearest")
            IC=I-I.mean(); TC=T-T.mean(); denom=max(float(IC@IC),1e-12)
            a=float(IC@TC/denom); b=float(T.mean()-a*I.mean())
            r=T-a*I-b; sse+=float(r@r); count+=len(T); gains.append([a,b])
            if derivative:
                Ix=ndimage.map_coordinates(cX,coords,order=3,prefilter=False,mode="nearest")*a
                Iy=ndimage.map_coordinates(cY,coords,order=3,prefilter=False,mode="nearest")*a
                J=np.empty((len(T),12)); J[:,0::2]=Ix[:,None]*B; J[:,1::2]=Iy[:,None]*B
                JC=J-J.mean(axis=0); v=JC.T@IC
                normal+=JC.T@JC-np.outer(v,v)/denom
                rhs+=JC.T@r-v*float(IC@r)/denom
        return normal,rhs,sse/count,gains

    def refine_coefficients(self,coeff,initial):
        p=np.asarray(initial,dtype=np.float64).copy()
        start_mse=self.evaluate(coeff,p,1,False)[2]; niter=0; condition=np.nan
        for stride,maxit in [(2,12),(1,10)]:
            for _ in range(maxit):
                H,g,mse,gains=self.evaluate(coeff,p,stride)
                condition=float(np.linalg.cond(H)); dp=np.linalg.solve(H+np.eye(12)*1e-8,g)
                maxdelta=max(float(np.max(np.linalg.norm(geom[2]@dp.reshape(6,2),axis=1))) for geom in self.geometry[stride])
                if maxdelta>0.5: dp*=0.5/maxdelta
                accepted=False
                for rate in [1.,0.5,0.25,0.125]:
                    candidate=p+rate*dp; newmse=self.evaluate(coeff,candidate,stride,False)[2]
                    if newmse <= mse+1e-10:
                        p=candidate; accepted=True; break
                niter+=1
                if not accepted or maxdelta*rate<2e-5: break
        final=self.evaluate(coeff,p,1,False)
        return dict(X=p,initial=np.asarray(initial),mse_initial=start_mse,mse_final=final[2],iterations=niter,condition=condition,gains=np.asarray(final[3]))

    def refine(self,current,initial):
        return self.refine_coefficients(self.coefficients(current),initial)

class DirectPairExtractor:
    def __init__(self,center,inset=10):
        self.center=Path(center)
        self.reference=read_gray(center)
        self.point_extractor=PairExtractor(center,method="gn17",include_screws=False)
        self.dense=DenseRegistration(self.reference,inset=inset)

    def extract(self,current):
        start=time.perf_counter(); pair=self.point_extractor.extract(current)
        valid=pair["valid"] & np.isfinite(pair["D"]).all(axis=1)
        points=pair["points"]
        if valid.sum()<12: raise ValueError("Insufficient valid QR points for P12 initialization")
        B=basis(points[:,0],points[:,1])
        initial=np.linalg.lstsq(B[valid],pair["D"][valid],rcond=None)[0].ravel()
        result=self.dense.refine(read_gray(current),initial)
        result.update(valid_points=int(valid.sum()),n_points=len(valid),pixel_count=self.dense.pixel_count,runtime_seconds=time.perf_counter()-start)
        return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--center",required=True);ap.add_argument("--current",required=True);ap.add_argument("--out",required=True);ap.add_argument("--inset",type=int,default=10);args=ap.parse_args()
    model=DirectPairExtractor(args.center,args.inset); result=model.extract(args.current)
    np.savez_compressed(args.out,**result)
    print(json.dumps({"output":str(Path(args.out).resolve()),"pixel_count":result['pixel_count'],"valid_points":result['valid_points'],"runtime_seconds":result['runtime_seconds'],"mse_final":result['mse_final']}))

if __name__=="__main__":main()

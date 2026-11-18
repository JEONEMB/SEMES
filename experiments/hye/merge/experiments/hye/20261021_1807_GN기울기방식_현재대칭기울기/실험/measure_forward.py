"""Current-image spline derivative GN and symmetric-gradient approximation."""
from pathlib import Path
import sys,json,time,concurrent.futures
import numpy as np,cv2
from scipy import ndimage
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B/'measurement'),str(B/'center_patch_v2_20261021')]
from spread import SpreadPairExtractor
from refine import Refiner
from measure import ROOT
from run import spatial
class ForwardRefiner(Refiner):
    def __init__(self,ref,points,window=17,symmetric=False):
        super().__init__(ref,points,window,.8);self.symmetric=symmetric
        self.T=self.template;self.phot=np.stack([self.T,np.ones_like(self.T)],axis=-1);self.photinv=np.linalg.pinv(self.phot)
        coeff=ndimage.spline_filter(cv2.GaussianBlur(ref.astype(float),(0,0),.8),order=3)
        self.grads=self.grad(coeff,self.coords)
    def sample(self,c,coords):return ndimage.map_coordinates(c,coords,order=3,prefilter=False,mode='nearest')
    def grad(self,c,coords):
        result=[]
        for axis in [1,0]:
            delta=np.zeros((2,1,1));delta[axis]=.01
            result.append((self.sample(c,coords+delta)-self.sample(c,coords-delta))/.02)
        return result
    def run(self,cur,initial):
        coeff=ndimage.spline_filter(cv2.GaussianBlur(cur.astype(float),(0,0),.8),order=3);d=initial.astype(float).copy()
        for it in range(20):
            coords=self.coords+d[:,::-1].T[:,:,None];values=self.sample(coeff,coords);photo=np.einsum('nkp,np->nk',self.photinv,values)
            residual=values-photo[:,0,None]*self.T-photo[:,1,None];gx,gy=self.grad(coeff,coords)
            if self.symmetric:gx=(gx+photo[:,0,None]*self.grads[0])/2;gy=(gy+photo[:,0,None]*self.grads[1])/2
            A=np.stack([gx,gy,-self.T,-np.ones_like(self.T)],axis=-1);scale=np.maximum(np.sqrt(np.mean(A*A,axis=1)),1e-8);A=A/scale[:,None,:]
            G=np.einsum('npi,npj->nij',A,A,optimize=True);rhs=np.einsum('npi,np->ni',A,residual,optimize=True)
            pars=np.linalg.solve(G+np.eye(4)[None]*1e-9,rhs[...,None])[...,0]/scale;step=np.clip(pars[:,:2],-.5,.5);d-=step
            if np.max(np.linalg.norm(step,axis=1))<2e-5:break
        q=np.c_[np.sqrt(np.mean(residual**2,axis=1)),np.linalg.norm(step,axis=1),photo[:,0],photo[:,1]];valid=np.isfinite(d).all(axis=1)&(np.linalg.norm(d-initial,axis=1)<1.5)&(q[:,1]<.02)
        return d.astype(np.float32),q.astype(np.float32),valid
def main():
    rng=np.random.default_rng(21);ref=cv2.GaussianBlur(rng.uniform(20,220,(96,96)),(0,0),1.2);yy,xx=np.mgrid[:96,:96];cur=ndimage.map_coordinates(ref,[yy-.12,xx-.23],order=3,mode='reflect');points=np.array([[25,25],[50,25],[25,50],[50,50]],np.float32);checks=[]
    for symmetric in [False,True]:
        r=ForwardRefiner(ref,points,17,symmetric).run(cur,np.zeros_like(points));err=float(abs(r[0]-[.23,.12]).max());assert err<.02;checks.append(dict(symmetric=symmetric,synthetic_max_error_px=err))
    (H/'numerical_checks.json').write_text(json.dumps(checks,indent=2));a=np.load(B/'center_patch_v2_20261021/tracks.npz');names=a['names'];pts=a['points'];audit=[]
    for window,symmetric in [(17,False),(31,False),(17,True),(31,True)]:
        key=f'forward{window}_symmetric{symmetric}';ext=SpreadPairExtractor(ROOT/'center.bmp',str(B/'center_patch_v2_20261021/features.npz'));start=time.time()
        for k,(x0,y0,x1,y1) in enumerate(ext.local_boxes):ext.models[k]=ForwardRefiner(ext.ref[y0:y1,x0:x1],ext.local_points[k],window,symmetric)
        paths=[ROOT/('center.bmp' if n=='center.bmp' else ('Y/' if n[0]=='Y' else 'Theta/')+n) for n in names];results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for i,r in enumerate(pool.map(ext.extract,paths)):
                results.append(r)
                if i%70==0:print(key,i,round(time.time()-start,1),flush=True)
        D=np.stack([r['D'] for r in results]).astype(float);D-=D[0];valid=np.stack([r['valid'] for r in results]);assert np.isfinite(D).all()
        np.savez_compressed(H/(key+'_tracks.npz'),names=names,points=pts,D=D,valid=valid,quality=np.stack([r['quality'] for r in results]));np.save(H/(key+'_features.npy'),spatial(pts,D));audit.append(dict(key=key,valid_fraction=float(valid.mean()),seconds=time.time()-start));print(audit[-1],flush=True)
    (H/'audit.json').write_text(json.dumps(dict(method='630 fixed IDs; spline derivative central difference eps .01px; translation with photometric gain/offset elimination, QR-only crop boundary extension; all finite points retained',results=audit),indent=2))
if __name__=='__main__':main()
